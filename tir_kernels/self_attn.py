from tvm.script import tir as T

@I.ir_module
class SelfAttention:
    @T.prim_func
    def sdpa(q: T.handle, k: T.handle, v: T.handle, sdpa_out: T.handle):
        N = T.int32()
        return q

    @T.prim_func
    def project_fused_qkv(
        n: T.int64, seq: T.int64, width: T.int64, 
        x: T.handle, 
        qkv_w: T.handle, qkv_b: T.handle, 
        out_q: T.handle, out_k: T.handle, out_v: T.handle 
    ):
        T.func_attr({"global_symbol": "project_fused_qkv", "tir.noalias": True})

        # We're assuming weights are packed here.
        X = T.match_buffer(x, [n, seq, width], "float32")
        QKV_W = T.match_buffer(qkv_w, [3*width, width], "float32")
        QKV_B = T.match_buffer(qkv_b, [3*width], "float32")
        OUT_Q = T.match_buffer(out_q, [n, seq, width], "float32")
        OUT_K = T.match_buffer(out_k, [n, seq, width], "float32")
        OUT_V = T.match_buffer(out_v, [n, seq, width], "float32")

        # Maybe keep x packed as well?
        # Basic matmul operation.
        # NxSEQxWIDTH @ WIDTHxWIDTH = NxSEQxWIDTH
        for _n, s, w, wk in T.grid(n, seq, width, width):
            with T.block('self_attn_qkv_w'):
                vn, vs, vw, vwk = T.axis.remap("SSSS", [_n, s, w, wk])
                OUT_Q[vn, vs, vw] += X[vn, vs, vwk] * QKV_W[vwk, vw]
                OUT_K[vn, vs, vw] += X[vn, vs, vwk] * QKV_W[1*width+vwk, vw]
                OUT_V[vn, vs, vw] += X[vn, vs, vwk] * QKV_W[2*width+vwk, vw]

        for n, s, w in T.grid(n, seq, width):
            with T.block('self_attn_qkv_b'):
                vn, vs, vw = T.axis.remap("SSS", [n, s, w])
                OUT_Q[vn, vs, vw] += QKV_B[vw]
                OUT_K[vn, vs, vw] += QKV_B[width + vw]
                OUT_V[vn, vs, vw] += QKV_B[2*width + vw]

    @R.function
    def main(
        x: R.Tensor(("n", "seq", "width"), dtype="float32"),
        qkv_w: R.Tensor(("3*width", "width"), dtype="float32"),
        qkv_b: R.Tensor(("3*width",), dtype="float32"),
        linear_w: R.Tensor(("width", "width"), dtype="float32"),
        linear_b: R.Tensor(("width",), dtype="float32")
    ):
        cls = SelfAttention 
        n, seq, width = T.int64(), T.int64(), T.int64()

        gv = R.call_tir(
            cls.project_fused_qkv, 
            args=[n, seq, width, x, qkv_w, qkv_b],
            out_sinfo=[
                R.Tensor((n,seq,width), "float32"),
                R.Tensor((n,seq,width), "float32"),
                R.Tensor((n,seq,width), "float32")
            ]
        )

        return gv