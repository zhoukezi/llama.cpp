#include "models.h"

ggml_cgraph * clip_graph_midashenglm::build() {
    const int n_mel_total = img.ny;
    const int n_mel_ac    = hparams.n_mel_bins_acoustic;
    const int n_mel       = n_mel_total - n_mel_ac;
    const int n_frames    = img.nx;

    const int stride_h = hparams.audio_patch_stride[0];
    const int stride_w = hparams.audio_patch_stride[1];

    const int n_freq_patches = n_mel / stride_h;
    const int n_time_patches = n_frames / stride_w;
    const int n_pos          = n_freq_patches * n_time_patches;

    const int subsample_k = hparams.proj_stack_factor;

    ggml_tensor * inp_full = build_inp_raw(1);
    ggml_tensor * inp;
    if (n_mel_ac > 0) {
        inp = ggml_view_2d(ctx0, inp_full, n_frames, n_mel, inp_full->nb[1], 0);
        inp = ggml_cont(ctx0, inp);
    } else {
        inp = ggml_reshape_2d(ctx0, inp_full, n_frames, n_mel);
    }

    ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    cur = ggml_mul(ctx0, cur, ggml_repeat(ctx0, model.init_bn_scale, cur));
    cur = ggml_add(ctx0, cur, ggml_repeat(ctx0, model.init_bn_shift, cur));

    inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cb(inp, "after_bn", -1);

    ggml_tensor * inp_4d = ggml_reshape_4d(ctx0, inp, n_frames, n_mel, 1, 1);

    cur = ggml_conv_2d(ctx0, model.a_patch_embd_w, inp_4d, stride_w, stride_h,  // stride
                       0, 0,                                                    // padding
                       1, 1);                                                   // dilation
    if (model.a_patch_embd_b) {
        cur = ggml_add(ctx0, cur, ggml_repeat(ctx0, ggml_reshape_4d(ctx0, model.a_patch_embd_b, 1, 1, n_embd, 1), cur));
    }
    cb(cur, "after_patch_embd", -1);

    cur = ggml_reshape_2d(ctx0, cur, n_pos, n_embd);

    const int n_pos_max = model.position_embeddings->ne[1];            // 252
    cur                 = ggml_cont(ctx0, ggml_transpose(ctx0, cur));  // → [n_embd, n_pos]

    if (n_pos <= n_pos_max) {
        // Simple case: positions fit within embedding table
        ggml_tensor * pos = ggml_view_2d(ctx0, model.position_embeddings, model.position_embeddings->ne[0], n_pos,
                                         model.position_embeddings->nb[1], 0);
        cur               = ggml_add(ctx0, cur, pos);
    } else {
        // Cycle position embeddings for each chunk of n_pos_max
        for (int offset = 0; offset < n_pos; offset += n_pos_max) {
            int           chunk_len = std::min(n_pos_max, n_pos - offset);
            ggml_tensor * pos       = ggml_view_2d(ctx0, model.position_embeddings, model.position_embeddings->ne[0],
                                                   chunk_len, model.position_embeddings->nb[1], 0);
            ggml_tensor * cur_chunk = ggml_view_2d(ctx0, cur, n_embd, chunk_len, cur->nb[1], offset * cur->nb[1]);
            ggml_tensor * sum       = ggml_add(ctx0, cur_chunk, pos);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, sum, cur_chunk));
        }
    }

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));  // → [n_pos, n_embd]
    cb(cur, "after_pos_embd", -1);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur = build_vit(cur, n_pos, NORM_TYPE_NORMAL, FFN_GELU, nullptr, nullptr);
    cb(cur, "after_transformer", -1);

    if (model.a_ac_patch_embd_w) {
        ggml_tensor * ac_inp = ggml_view_2d(ctx0, inp_full, n_frames, n_mel_ac, inp_full->nb[1], n_mel * sizeof(float));
        ac_inp               = ggml_cont(ctx0, ac_inp);

        ggml_tensor * ac_4d  = ggml_reshape_4d(ctx0, ac_inp, n_frames, n_mel_ac, 1, 1);
        ggml_tensor * ac_cur = ggml_conv_2d(ctx0, model.a_ac_patch_embd_w, ac_4d, stride_w, n_mel_ac, 0, 0, 1, 1);
        if (model.a_ac_patch_embd_b) {
            ac_cur =
                ggml_add(ctx0, ac_cur,
                         ggml_repeat(ctx0, ggml_reshape_4d(ctx0, model.a_ac_patch_embd_b, 1, 1, n_embd, 1), ac_cur));
        }

        int n_ac_tokens = n_frames / stride_w;  // same as n_time_patches
        ac_cur          = ggml_reshape_2d(ctx0, ac_cur, n_ac_tokens, n_embd);
        ac_cur          = ggml_cont(ctx0, ggml_transpose(ctx0, ac_cur));
        ac_cur          = build_norm(ac_cur, model.a_ac_post_ln_w, model.a_ac_post_ln_b, NORM_TYPE_NORMAL, eps, -1);
        cb(ac_cur, "acoustic_branch", -1);

        if (n_pos > n_ac_tokens) {
            cur = ggml_view_2d(ctx0, cur, n_embd, n_ac_tokens, cur->nb[1], 0);
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_add(ctx0, cur, ac_cur);
        cb(cur, "after_acoustic_add", -1);
    }

    int n_tokens = (int) cur->ne[1];
    int discard  = n_tokens % subsample_k;
    if (discard > 0) {
        n_tokens -= discard;
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
        cur = ggml_cont(ctx0, cur);
    }

    int n_out_tokens = n_tokens / subsample_k;
    cur              = ggml_reshape_2d(ctx0, cur, n_embd * subsample_k, n_out_tokens);
    cb(cur, "after_subsample", -1);

    cur = build_mm(model.mm_0_w, cur);
    if (model.mm_0_b) {
        cur = ggml_add(ctx0, cur, ggml_repeat(ctx0, model.mm_0_b, cur));
    }
    cur = ggml_gelu(ctx0, cur);
    cb(cur, "after_proj_gelu", -1);

    cur = build_mm(model.mm_2_w, cur);
    if (model.mm_2_b) {
        cur = ggml_add(ctx0, cur, ggml_repeat(ctx0, model.mm_2_b, cur));
    }
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
