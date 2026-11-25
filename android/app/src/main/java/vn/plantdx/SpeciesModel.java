package vn.plantdx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import android.content.Context;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;

/** Một mô hình theo loài: assets/<loài>/model.onnx (INT8) + meta.json (tên lớp, T*, τ_E, τ_c, tiền xử lý). */
public final class SpeciesModel implements AutoCloseable {
    public static final class Prediction {
        public final SpeciesModel model;
        public final Preprocess.Status status;
        public final int best;
        public final double[] probs;
        public final double energy;
        public final long millis;

        Prediction(SpeciesModel model, Preprocess.Status status, int best, double[] probs, double energy, long millis) {
            this.model = model; this.status = status; this.best = best;
            this.probs = probs; this.energy = energy; this.millis = millis;
        }

        /** Năng lượng vượt ngưỡng bao nhiêu; dùng để tự chọn loài (như auto_predict trong inference.py). */
        public double margin() { return energy - model.tauEnergy; }
    }

    public final String species, nameVi;
    public final String[] namesVi;
    public final int resize, size;
    public final double T, tauEnergy, tauConf;
    private final float[] mean, std;
    private final OrtEnvironment env;
    private final OrtSession session;

    public SpeciesModel(Context ctx, String species, String nameVi) throws IOException, OrtException, JSONException {
        this.species = species;
        this.nameVi = nameVi;
        JSONObject m = new JSONObject(new String(readAll(ctx.getAssets().open(species + "/meta.json")), StandardCharsets.UTF_8));
        JSONArray names = m.getJSONArray("names_vi");
        namesVi = new String[names.length()];
        for (int i = 0; i < namesVi.length; i++) namesVi[i] = names.getString(i);
        resize = m.getInt("resize");
        size = m.getInt("img_size");
        T = m.getDouble("T");
        tauEnergy = m.getDouble("tau_energy");
        tauConf = m.getDouble("tau_conf");
        mean = floats(m.getJSONArray("mean"));
        std = floats(m.getJSONArray("std"));

        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(4);
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        session = env.createSession(readAll(ctx.getAssets().open(species + "/model.onnx")), opts);
    }

    /** Tên bệnh không kèm tiền tố loài: "Cà chua – Bệnh úa sớm" -> "Bệnh úa sớm". */
    public String diseaseName(int k) {
        String s = namesVi[k];
        int i = s.indexOf(" – ");
        return i >= 0 ? s.substring(i + 3) : s;
    }

    /** Ảnh ARGB w×h đã đưa cạnh ngắn về {@link #resize} (xem {@link Resample}); hàm tự cắt giữa size×size. */
    public Prediction predict(int[] resized, int w, int h) throws OrtException {
        int[] off = Preprocess.cropOffset(w, h, size);
        int[] px = new int[size * size];
        for (int y = 0; y < size; y++) System.arraycopy(resized, (y + off[1]) * w + off[0], px, y * size, size);
        float[] chw = Preprocess.toChw(px, size, mean, std);

        long t0 = System.nanoTime();
        try (OnnxTensor x = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1, 3, size, size});
             OrtSession.Result out = session.run(Collections.singletonMap("input", x))) {
            float[] logits = ((float[][]) out.get(0).getValue())[0];
            long ms = (System.nanoTime() - t0) / 1_000_000;
            double[] p = Preprocess.calibratedProbs(logits, T);
            double e = Preprocess.energy(logits, T);
            int k = Preprocess.topK(p, 1)[0];
            return new Prediction(this, Preprocess.decide(e, p[k], tauEnergy, tauConf), k, p, e, ms);
        }
    }

    @Override public void close() throws OrtException { session.close(); }

    private static float[] floats(JSONArray a) throws JSONException {
        float[] f = new float[a.length()];
        for (int i = 0; i < f.length; i++) f[i] = (float) a.getDouble(i);
        return f;
    }

    private static byte[] readAll(InputStream in) throws IOException {
        try (in; ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0; ) bos.write(buf, 0, n);
            return bos.toByteArray();
        }
    }
}
