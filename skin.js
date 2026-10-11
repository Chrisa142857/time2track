// UmeTrack hand skinning, ported from emg2pose (hand_skinning.py): Rodrigues joint rotations about fixed axes,
// 4 joints chained per finger into 17 bone frames (root, wrist, 3 per finger), then linear blend skinning
function makeSkinner(P) {
  const rest = P.rest, axes = P.axes, V = P.verts.length / 3, NB = 17, W = P.weights, verts = P.verts;
  const L = new Float64Array(20 * 12), F = new Float64Array(NB * 12);
  const mul = (A, a, B, b, out, o) => {           // 3x4 affine: out = A * B
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) out[o + 4 * r + c] = A[a + 4 * r] * B[b + c] + A[a + 4 * r + 1] * B[b + 4 + c] + A[a + 4 * r + 2] * B[b + 8 + c];
      out[o + 4 * r + 3] = A[a + 4 * r] * B[b + 3] + A[a + 4 * r + 1] * B[b + 7] + A[a + 4 * r + 2] * B[b + 11] + A[a + 4 * r + 3];
    }
  };
  const tmp = new Float64Array(12);
  return function skin(ang, out) {
    for (let k = 0; k < 20; k++) {
      const wx = axes[3 * k] * ang[k], wy = axes[3 * k + 1] * ang[k], wz = axes[3 * k + 2] * ang[k];
      const th = Math.hypot(wx, wy, wz), o = 12 * k;
      let r00 = 1, r01 = 0, r02 = 0, r10 = 0, r11 = 1, r12 = 0, r20 = 0, r21 = 0, r22 = 1;
      if (th > 1e-12) {
        const x = wx / th, y = wy / th, z = wz / th, s = Math.sin(th), c1 = 1 - Math.cos(th);
        r00 = 1 + c1 * (-y * y - z * z); r01 = -s * z + c1 * x * y; r02 = s * y + c1 * x * z;
        r10 = s * z + c1 * x * y; r11 = 1 + c1 * (-x * x - z * z); r12 = -s * x + c1 * y * z;
        r20 = -s * y + c1 * x * z; r21 = s * x + c1 * y * z; r22 = 1 + c1 * (-x * x - y * y);
      }
      const px = rest[3 * k], py = rest[3 * k + 1], pz = rest[3 * k + 2];
      L[o] = r00; L[o + 1] = r01; L[o + 2] = r02; L[o + 3] = px - (r00 * px + r01 * py + r02 * pz);
      L[o + 4] = r10; L[o + 5] = r11; L[o + 6] = r12; L[o + 7] = py - (r10 * px + r11 * py + r12 * pz);
      L[o + 8] = r20; L[o + 9] = r21; L[o + 10] = r22; L[o + 11] = pz - (r20 * px + r21 * py + r22 * pz);
    }
    const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
    for (let q = 0; q < 12; q++) { F[q] = I[q]; F[12 + q] = I[q]; }
    for (let f = 0; f < 5; f++) {                   // M1 = L0, M2 = M1 L1, M3 = M2 L2, M4 = M3 L3; frames M2, M3, M4
      const b = 2 + 3 * f, j = 4 * f;
      mul(L, 12 * j, L, 12 * (j + 1), F, 12 * b);
      mul(F, 12 * b, L, 12 * (j + 2), F, 12 * (b + 1));
      mul(F, 12 * (b + 1), L, 12 * (j + 3), F, 12 * (b + 2));
    }
    for (let v = 0; v < V; v++) {
      const x = verts[3 * v], y = verts[3 * v + 1], z = verts[3 * v + 2];
      let ox = 0, oy = 0, oz = 0;
      for (let bn = 0; bn < NB; bn++) {
        const w = W[v * NB + bn];
        if (!w) continue;
        const o = 12 * bn;
        ox += w * (F[o] * x + F[o + 1] * y + F[o + 2] * z + F[o + 3]);
        oy += w * (F[o + 4] * x + F[o + 5] * y + F[o + 6] * z + F[o + 7]);
        oz += w * (F[o + 8] * x + F[o + 9] * y + F[o + 10] * z + F[o + 11]);
      }
      out[3 * v] = ox; out[3 * v + 1] = oy; out[3 * v + 2] = oz;
    }
    return out;
  };
}
if (typeof module !== 'undefined') module.exports = makeSkinner;
