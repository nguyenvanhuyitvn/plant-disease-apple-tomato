package vn.plantdx;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.content.FileProvider;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int REQ_CAMERA = 1, REQ_GALLERY = 2;
    private static final int DISPLAY_MAX = 1280;
    private static final int MODEL_MAX = 2048;   // ảnh camera 12 MP được thu về cạnh dài 2.048 trước khi resize kiểu Pillow

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private volatile SpeciesModel apple, tomato;
    private Uri cameraUri, lastUri;

    private ImageView image;
    private TextView placeholder, label, conf, species, notice, details;
    private LinearLayout card, topk;
    private RadioGroup speciesGroup;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_main);
        image = findViewById(R.id.image);
        placeholder = findViewById(R.id.placeholder);
        label = findViewById(R.id.result_label);
        conf = findViewById(R.id.result_conf);
        species = findViewById(R.id.result_species);
        notice = findViewById(R.id.notice);
        details = findViewById(R.id.details);
        card = findViewById(R.id.result_card);
        topk = findViewById(R.id.topk);
        speciesGroup = findViewById(R.id.species_group);
        if (state != null && state.getString("cameraUri") != null) cameraUri = Uri.parse(state.getString("cameraUri"));

        findViewById(R.id.btn_camera).setOnClickListener(v -> openCamera());
        findViewById(R.id.btn_gallery).setOnClickListener(v -> {
            Intent i = new Intent(Intent.ACTION_GET_CONTENT).setType("image/*");
            startActivityForResult(Intent.createChooser(i, getString(R.string.pick_photo)), REQ_GALLERY);
        });
        // Đổi loài thì phân tích lại ảnh đang xem.
        speciesGroup.setOnCheckedChangeListener((g, id) -> { if (lastUri != null) analyze(lastUri); });

        // Nạp hai mô hình ở luồng nền để không làm đơ giao diện.
        worker.execute(() -> {
            try {
                apple = new SpeciesModel(this, "apple", "Táo");
                tomato = new SpeciesModel(this, "tomato", "Cà chua");
            } catch (Exception e) {
                toast("Không nạp được mô hình: " + e.getMessage());
            }
        });
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        if (cameraUri != null) out.putString("cameraUri", cameraUri.toString());
    }

    private void openCamera() {
        try {
            File dir = new File(getCacheDir(), "photos");
            dir.mkdirs();
            File f = new File(dir, "capture.jpg");
            cameraUri = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", f);
            Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                    .putExtra(MediaStore.EXTRA_OUTPUT, cameraUri)
                    .addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivityForResult(i, REQ_CAMERA);
        } catch (ActivityNotFoundException e) {
            toast("Không tìm thấy ứng dụng camera");
        }
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (res != RESULT_OK) return;
        Uri uri = req == REQ_CAMERA ? cameraUri : (data != null ? data.getData() : null);
        if (uri != null) analyze(uri);
    }

    /** Các mô hình cần chạy theo lựa chọn loài; rỗng nếu chưa nạp xong. */
    private List<SpeciesModel> selectedModels() {
        List<SpeciesModel> out = new ArrayList<>();
        int id = speciesGroup.getCheckedRadioButtonId();
        if (apple == null || tomato == null) return out;
        if (id == R.id.sp_apple) out.add(apple);
        else if (id == R.id.sp_tomato) out.add(tomato);
        else { out.add(apple); out.add(tomato); }
        return out;
    }

    private void analyze(Uri uri) {
        lastUri = uri;
        card.setVisibility(View.VISIBLE);
        label.setText(R.string.analyzing);
        label.setTextColor(getColor(R.color.ink));
        conf.setText("");
        species.setVisibility(View.GONE);
        notice.setVisibility(View.GONE);
        details.setText("");
        topk.removeAllViews();
        List<SpeciesModel> models = selectedModels();
        boolean auto = models.size() > 1;

        worker.execute(() -> {
            try {
                if (models.isEmpty()) {
                    // Mô hình được nạp trong cùng hàng đợi; tới đây là đã nạp xong hoặc đã lỗi.
                    if (apple != null && tomato != null) ui.post(() -> analyze(uri));
                    else toast("Mô hình chưa sẵn sàng");
                    return;
                }
                Bitmap display = decode(uri, DISPLAY_MAX);
                ui.post(() -> { image.setImageBitmap(display); placeholder.setVisibility(View.GONE); });
                // Ảnh gốc -> cạnh ngắn 255 bằng đúng phép bilinear của Pillow (Resample), như app/inference.py.
                Bitmap full = decode(uri, MODEL_MAX);
                int w = full.getWidth(), h = full.getHeight();
                int[] src = new int[w * h];
                full.getPixels(src, 0, w, 0, 0, w, h);
                full.recycle();
                int[] t = Preprocess.resizedSize(w, h, models.get(0).resize);   // hai mô hình cùng resize = 255
                int[] small = Resample.resize(src, w, h, t[0], t[1]);
                SpeciesModel.Prediction best = null;
                for (SpeciesModel m : models) {
                    SpeciesModel.Prediction p = m.predict(small, t[0], t[1]);
                    if (best == null || p.margin() > best.margin()) best = p;
                }
                SpeciesModel.Prediction r = best;
                ui.post(() -> show(r, auto));
            } catch (Exception e) {
                toast("Không đọc được ảnh: " + e.getMessage());
            }
        });
    }

    /**
     * Giải mã ảnh (tự xoay theo EXIF như ImageOps.exif_transpose). Ảnh có cạnh dài hơn maxSide được
     * bộ giải mã thu nhỏ trước cho đỡ tốn bộ nhớ; ảnh từ tập dữ liệu (cạnh ≤ 2.048 px) giữ nguyên kích thước.
     */
    private Bitmap decode(Uri uri, int maxSide) throws java.io.IOException {
        ImageDecoder.Source src = ImageDecoder.createSource(getContentResolver(), uri);
        return ImageDecoder.decodeBitmap(src, (decoder, info, s) -> {
            int w = info.getSize().getWidth(), h = info.getSize().getHeight();
            float sc = Math.min(1f, (float) maxSide / Math.max(w, h));
            if (sc < 1f) decoder.setTargetSize(Math.max(1, Math.round(w * sc)), Math.max(1, Math.round(h * sc)));
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);   // để đọc được pixel
        });
    }

    private void show(SpeciesModel.Prediction r, boolean auto) {
        SpeciesModel m = r.model;
        double pBest = r.probs[r.best];
        String name = m.diseaseName(r.best);

        switch (r.status) {
            case NOT_LEAF:
                label.setText(auto ? getString(R.string.not_leaf_any) : getString(R.string.not_leaf_one, m.nameVi.toLowerCase(Locale.ROOT)));
                label.setTextColor(getColor(R.color.red_text));
                notice.setText(R.string.not_leaf_hint);
                notice.setBackgroundResource(R.drawable.err_bg);
                notice.setTextColor(getColor(R.color.red_text));
                notice.setVisibility(View.VISIBLE);
                break;
            case UNCERTAIN:
                label.setText(getString(R.string.maybe, name));
                label.setTextColor(getColor(R.color.amber_text));
                notice.setText(R.string.uncertain);
                notice.setBackgroundResource(R.drawable.warn_bg);
                notice.setTextColor(getColor(R.color.amber_text));
                notice.setVisibility(View.VISIBLE);
                break;
            default:
                label.setText(name);
                label.setTextColor(getColor(R.color.green_dark));
                notice.setVisibility(View.GONE);
        }
        conf.setText(pct(String.format(Locale.US, "Độ tin cậy (đã hiệu chỉnh) %.0f%% · xử lý %d ms", pBest * 100, r.millis)));
        species.setText(getString(auto ? R.string.species_auto : R.string.species_fixed, m.nameVi));
        species.setVisibility(View.VISIBLE);
        details.setText(pct(String.format(Locale.US, "Năng lượng %.2f (ngưỡng τ_E %.2f) · T* = %.2f · τ_c = %.2f",
                r.energy, m.tauEnergy, m.T, m.tauConf)));

        topk.removeAllViews();
        for (int k : Preprocess.topK(r.probs, 3)) topk.addView(row(m.diseaseName(k), r.probs[k]));
    }

    private static String pct(String s) { return s.replace('.', ','); }

    private View row(String name, double p) {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(0, dp(10), 0, 0);

        TextView t = new TextView(this);
        t.setText(String.format(Locale.US, "%s — %.0f%%", name, p * 100));
        t.setTextColor(getColor(R.color.ink));
        t.setTextSize(14);
        box.addView(t);

        ProgressBar bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(1000);
        bar.setProgress((int) Math.round(p * 1000));
        bar.setProgressTintList(ColorStateList.valueOf(getColor(R.color.green)));
        box.addView(bar, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(8)));
        return box;
    }

    private int dp(int v) { return Math.round(v * getResources().getDisplayMetrics().density); }

    private void toast(String msg) { ui.post(() -> Toast.makeText(this, msg, Toast.LENGTH_LONG).show()); }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        worker.execute(() -> {
            try { if (apple != null) apple.close(); if (tomato != null) tomato.close(); } catch (Exception ignored) {}
        });
        worker.shutdown();
    }
}
