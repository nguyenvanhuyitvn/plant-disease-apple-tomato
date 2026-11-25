package vn.plantdx;

/**
 * Thu nhỏ/phóng ảnh đúng như Pillow {@code Image.resize(size, Image.BILINEAR)} (thư viện C
 * libImaging/Resample.c): bộ lọc tam giác có khử răng cưa khi thu nhỏ, hệ số nguyên 22 bit,
 * làm hai lượt ngang rồi dọc, ảnh trung gian 8 bit.
 *
 * Lý do không dùng bộ giải mã của Android để resize: mô hình cà chua nhạy với cách resize
 * (đổi sang LANCZOS/BOX/BICUBIC thì 15–25% ảnh test đổi kết luận), nên ứng dụng phải
 * tiền xử lý giống hệt lúc đánh giá (app/inference.py).
 */
public final class Resample {
    private static final int PRECISION_BITS = 32 - 8 - 2;

    private Resample() {}

    private static double bilinear(double x) {
        if (x < 0.0) x = -x;
        return x < 1.0 ? 1.0 - x : 0.0;
    }

    /** Hệ số cho một chiều: bounds[2*i] = vị trí bắt đầu, bounds[2*i+1] = số điểm; kk[i*ksize + j]. */
    private static final class Coeffs {
        int ksize;
        int[] bounds;
        int[] kk;
    }

    private static Coeffs coeffs(int inSize, int outSize) {
        double support = 1.0;   // bộ lọc bilinear
        double scale = (double) inSize / outSize;
        double filterscale = Math.max(scale, 1.0);
        support *= filterscale;
        int ksize = (int) Math.ceil(support) * 2 + 1;
        double[] k = new double[outSize * ksize];
        int[] bounds = new int[outSize * 2];
        double ss = 1.0 / filterscale;
        for (int xx = 0; xx < outSize; xx++) {
            double center = (xx + 0.5) * scale;
            double ww = 0.0;
            int xmin = (int) (center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = (int) (center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            xmax -= xmin;
            for (int x = 0; x < xmax; x++) {
                double w = bilinear((x + xmin - center + 0.5) * ss);
                k[xx * ksize + x] = w;
                ww += w;
            }
            for (int x = 0; x < xmax; x++) if (ww != 0.0) k[xx * ksize + x] /= ww;
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        Coeffs c = new Coeffs();
        c.ksize = ksize;
        c.bounds = bounds;
        c.kk = new int[k.length];
        for (int i = 0; i < k.length; i++) {
            // normalize_coeffs_8bpc: (int)(0.5 + k * 2^22), số âm làm tròn về phía âm
            double v = k[i] * (1 << PRECISION_BITS);
            c.kk[i] = (int) (k[i] < 0 ? -0.5 + v : 0.5 + v);
        }
        return c;
    }

    private static int clip8(int ss) {
        int v = ss >> PRECISION_BITS;
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }

    /** Ảnh ARGB (w×h) -> ảnh ARGB (nw×nh). Kênh alpha đặt 255. */
    public static int[] resize(int[] src, int w, int h, int nw, int nh) {
        int[] cur = src;
        int cw = w;
        if (nw != w) {   // lượt ngang
            Coeffs c = coeffs(w, nw);
            int[] out = new int[nw * h];
            for (int y = 0; y < h; y++) {
                int row = y * w;
                for (int xx = 0; xx < nw; xx++) {
                    int xmin = c.bounds[xx * 2], xmax = c.bounds[xx * 2 + 1], o = xx * c.ksize;
                    int r = 1 << (PRECISION_BITS - 1), g = r, b = r;
                    for (int x = 0; x < xmax; x++) {
                        int p = cur[row + x + xmin], kv = c.kk[o + x];
                        r += ((p >> 16) & 0xFF) * kv;
                        g += ((p >> 8) & 0xFF) * kv;
                        b += (p & 0xFF) * kv;
                    }
                    out[y * nw + xx] = 0xFF000000 | (clip8(r) << 16) | (clip8(g) << 8) | clip8(b);
                }
            }
            cur = out;
            cw = nw;
        }
        if (nh != h) {   // lượt dọc
            Coeffs c = coeffs(h, nh);
            int[] out = new int[cw * nh];
            for (int yy = 0; yy < nh; yy++) {
                int ymin = c.bounds[yy * 2], ymax = c.bounds[yy * 2 + 1], o = yy * c.ksize;
                for (int x = 0; x < cw; x++) {
                    int r = 1 << (PRECISION_BITS - 1), g = r, b = r;
                    for (int y = 0; y < ymax; y++) {
                        int p = cur[(y + ymin) * cw + x], kv = c.kk[o + y];
                        r += ((p >> 16) & 0xFF) * kv;
                        g += ((p >> 8) & 0xFF) * kv;
                        b += (p & 0xFF) * kv;
                    }
                    out[yy * cw + x] = 0xFF000000 | (clip8(r) << 16) | (clip8(g) << 8) | clip8(b);
                }
            }
            cur = out;
        }
        return cur;
    }
}
