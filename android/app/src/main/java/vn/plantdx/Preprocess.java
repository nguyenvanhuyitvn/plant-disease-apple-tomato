package vn.plantdx;

/**
 * Phần tính toán thuần Java (không phụ thuộc Android), khớp với app/inference.py:
 *   - tiền xử lý: cạnh ngắn -> resize (255), cắt giữa size (224), chuẩn hoá theo mean/std trong meta.json
 *   - hiệu chỉnh: xác suất = softmax(logits / T), năng lượng = T · logsumexp(logits / T)
 *   - quyết định: năng lượng < τ_E -> NOT_LEAF; xác suất lớn nhất < τ_c -> UNCERTAIN; còn lại OK
 */
public final class Preprocess {
    public enum Status { OK, UNCERTAIN, NOT_LEAF }

    private Preprocess() {}

    /** Kích thước sau khi đưa cạnh ngắn về {@code resize}, giữ tỉ lệ (giống inference.py). */
    public static int[] resizedSize(int w, int h, int resize) {
        if (w <= h) return new int[]{resize, (int) ((long) resize * h / w)};
        return new int[]{(int) ((long) resize * w / h), resize};
    }

    /**
     * Góc trên-trái của vùng cắt giữa size×size. Python round() (inference.py, torchvision CenterCrop)
     * làm tròn nửa về số chẵn: round(14.5) = 14; Math.round cho 15 nên dùng Math.rint.
     */
    public static int[] cropOffset(int w, int h, int size) {
        return new int[]{(int) Math.rint((w - size) / 2.0), (int) Math.rint((h - size) / 2.0)};
    }

    /** Pixel ARGB (size×size, theo hàng) -> tensor CHW float đã chuẩn hoá. */
    public static float[] toChw(int[] argb, int size, float[] mean, float[] std) {
        int n = size * size;
        float[] out = new float[3 * n];
        for (int i = 0; i < n; i++) {
            int p = argb[i];
            out[i] = (((p >> 16) & 0xFF) / 255f - mean[0]) / std[0];
            out[n + i] = (((p >> 8) & 0xFF) / 255f - mean[1]) / std[1];
            out[2 * n + i] = ((p & 0xFF) / 255f - mean[2]) / std[2];
        }
        return out;
    }

    public static double logSumExp(double[] z) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : z) m = Math.max(m, v);
        double s = 0;
        for (double v : z) s += Math.exp(v - m);
        return m + Math.log(s);
    }

    /** Xác suất đã hiệu chỉnh nhiệt độ. */
    public static double[] calibratedProbs(float[] logits, double T) {
        double[] z = new double[logits.length];
        for (int i = 0; i < z.length; i++) z[i] = logits[i] / T;
        double lse = logSumExp(z);
        double[] p = new double[z.length];
        for (int i = 0; i < p.length; i++) p[i] = Math.exp(z[i] - lse);
        return p;
    }

    /** Điểm năng lượng S(x) = T · logsumexp(logits / T). */
    public static double energy(float[] logits, double T) {
        double[] z = new double[logits.length];
        for (int i = 0; i < z.length; i++) z[i] = logits[i] / T;
        return T * logSumExp(z);
    }

    public static Status decide(double energy, double pMax, double tauEnergy, double tauConf) {
        if (energy < tauEnergy) return Status.NOT_LEAF;
        if (pMax < tauConf) return Status.UNCERTAIN;
        return Status.OK;
    }

    /** Chỉ số k lớp có xác suất cao nhất, giảm dần. */
    public static int[] topK(double[] p, int k) {
        Integer[] idx = new Integer[p.length];
        for (int i = 0; i < p.length; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(p[b], p[a]));
        int[] out = new int[Math.min(k, p.length)];
        for (int i = 0; i < out.length; i++) out[i] = idx[i];
        return out;
    }
}
