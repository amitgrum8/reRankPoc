run 2026-11-14_sasrec | encoder Qwen/Qwen3-Embedding-8B | alpha 0.1 beta 0.1 gamma 1.0

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R10_mix | R11_mix_crit | R12_mix_reviews | R13_near | R13_near_ctrl | A1_likes_only | A2_dis_all | A3_dis_good | A4_mixed | A5_coral | A6_satisfaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 |
| in_list_mean_rank_lightgcn | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 |
| in_list_mean_rank | 167.5 | 163.8 | 166.5 | 164.5 | 165.1 | 163.0 | 166.6 | 166.2 | 157.0 | 156.2 | 157.8 | 158.0 | 158.3 | 155.2 | 158.0 | 159.0 | 156.9 | 163.2 | 158.6 |
| in_list_rank_change | 0.0 | -3.7 | -1.0 | -2.9 | -2.4 | -4.5 | -0.9 | -1.2 | -10.5 | -11.3 | -9.7 | -9.5 | -9.2 | -12.3 | -9.4 | -8.5 | -10.5 | -4.3 | -8.9 |
| in_list_moved_up | 0 | 288 | 258 | 285 | 277 | 302 | 342 | 207 | 300 | 301 | 304 | 303 | 305 | 300 | 300 | 291 | 299 | 305 | 302 |
| in_list_stayed | 543 | 42 | 47 | 45 | 46 | 47 | 64 | 58 | 18 | 22 | 16 | 18 | 17 | 17 | 22 | 13 | 15 | 21 | 19 |
| in_list_moved_down | 0 | 213 | 238 | 213 | 220 | 194 | 137 | 278 | 225 | 220 | 223 | 222 | 221 | 226 | 221 | 239 | 229 | 217 | 222 |
| in_list_top10 | 41 | 41 | 46 | 41 | 46 | 41 | 40 | 44 | 47 | 48 | 50 | 51 | 48 | 51 | 49 | 48 | 48 | 45 | 47 |
| out_of_list | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 |
| out_pulled_in | 0 | 31 | 36 | 30 | 40 | 38 | 42 | 28 | 264 | 286 | 286 | 258 | 269 | 350 | 378 | 1157 | 514 | 137 | 231 |
| all_mean_rank | 429.1 | 428.1 | 428.7 | 428.3 | 428.4 | 427.9 | 428.7 | 428.6 | 422.7 | 422.6 | 422.7 | 423.1 | 423.2 | 420.4 | 421.6 | 417.7 | 417.9 | 426.7 | 424.3 |
| in_list_change_liked | 0.0 | -4.2 | -1.7 | -3.6 | -3.5 | -5.3 | -1.9 | -1.4 | -11.8 | -12.3 | -6.9 | -6.3 | -7.4 | -10.4 | -6.3 | -3.5 | -6.8 | -2.1 | -7.6 |
| in_list_change_disliked | 0.0 | -3.2 | -0.2 | -2.3 | 0.2 | -3.2 | 1.7 | -1.3 | -8.2 | -8.9 | -15.6 | -15.7 | -11.4 | -18.2 | -15.4 | -19.5 | -16.4 | -9.0 | -12.2 |

mean rank change of the in-list targets per user, bootstrap over users (95% interval):

| reranker | mean change | 2.5% | 97.5% |
|---|---|---|---|
| R1_like | -4.4 | -6.6 | -2.3 |
| R2_like-dis | -1.9 | -4.3 | +0.3 |
| R3_soft | -3.6 | -5.8 | -1.6 |
| R4_rocchio | -2.4 | -4.5 | -0.5 |
| R5_naive | -4.7 | -6.7 | -2.7 |
| R8_rev_like | -0.2 | -2.1 | +2.0 |
| R9_rev | -0.4 | -2.3 | +1.4 |
| R10_mix | -12.7 | -20.3 | -5.2 |
| R11_mix_crit | -14.7 | -22.1 | -7.5 |
| R12_mix_reviews | -11.2 | -17.6 | -5.1 |
| R13_near | -10.8 | -16.8 | -4.7 |
| R13_near_ctrl | -10.4 | -16.6 | -4.5 |
| A1_likes_only | -12.3 | -20.0 | -5.3 |
| A2_dis_all | -12.6 | -20.4 | -5.5 |
| A3_dis_good | -11.3 | -21.2 | -1.5 |
| A4_mixed | -14.7 | -24.3 | -5.1 |
| A5_coral | -5.3 | -10.1 | -1.0 |
| A6_satisfaction | -9.3 | -14.6 | -4.2 |

control: the same rerankers with another user's persona (what any persona moves):

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R10_mix | R11_mix_crit | R12_mix_reviews | R13_near | A1_likes_only | A2_dis_all | A3_dis_good | A4_mixed | A5_coral | A6_satisfaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 |
| in_list_mean_rank_lightgcn | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 |
| in_list_mean_rank | 167.5 | 168.7 | 167.8 | 168.5 | 168.4 | 168.3 | 168.0 | 168.0 | 174.5 | 172.9 | 171.2 | 170.7 | 171.1 | 172.5 | 173.9 | 173.0 | 165.0 | 170.8 |
| in_list_rank_change | 0.0 | 1.2 | 0.4 | 1.0 | 0.9 | 0.8 | 0.5 | 0.5 | 7.0 | 5.4 | 3.7 | 3.2 | 3.6 | 5.0 | 6.4 | 5.5 | -2.5 | 3.3 |
| in_list_moved_up | 0 | 209 | 254 | 211 | 243 | 226 | 336 | 185 | 217 | 233 | 249 | 241 | 239 | 242 | 244 | 232 | 282 | 246 |
| in_list_stayed | 543 | 50 | 53 | 50 | 49 | 52 | 66 | 61 | 19 | 16 | 17 | 24 | 14 | 18 | 12 | 11 | 25 | 21 |
| in_list_moved_down | 0 | 284 | 236 | 282 | 251 | 265 | 141 | 297 | 307 | 294 | 277 | 278 | 290 | 283 | 287 | 300 | 236 | 276 |
| in_list_top10 | 41 | 40 | 43 | 40 | 42 | 40 | 39 | 41 | 36 | 36 | 37 | 39 | 43 | 40 | 37 | 37 | 45 | 39 |
| out_of_list | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 |
| out_pulled_in | 0 | 29 | 40 | 29 | 37 | 30 | 38 | 31 | 250 | 237 | 282 | 252 | 338 | 364 | 1159 | 496 | 138 | 186 |
| all_mean_rank | 429.1 | 429.2 | 429.0 | 429.2 | 429.1 | 429.1 | 429.1 | 429.1 | 427.9 | 427.5 | 427.1 | 427.2 | 425.7 | 426.3 | 422.5 | 422.9 | 427.3 | 428.0 |
| in_list_change_liked | 0.0 | 0.8 | -1.0 | 0.6 | -0.9 | 0.3 | -0.3 | 0.0 | 6.2 | 4.6 | 5.5 | 4.7 | 6.0 | 7.4 | 11.9 | 9.8 | -0.3 | 3.8 |
| in_list_change_disliked | 0.0 | 1.4 | 3.1 | 1.5 | 3.4 | -0.3 | 2.3 | 0.9 | 4.5 | 3.9 | -1.5 | -1.9 | -5.4 | 0.6 | -4.0 | 0.6 | -8.3 | -0.2 |

| reranker | in-list rank change, own persona | another user's persona |
|---|---|---|
| R1_like | -3.7 | +1.2 |
| R2_like-dis | -1.0 | +0.4 |
| R3_soft | -2.9 | +1.0 |
| R4_rocchio | -2.4 | +0.9 |
| R5_naive | -4.5 | +0.8 |
| R8_rev_like | -0.9 | +0.5 |
| R9_rev | -1.2 | +0.5 |
| R10_mix | -10.5 | +7.0 |
| R11_mix_crit | -11.3 | +5.4 |
| R12_mix_reviews | -9.7 | +3.7 |
| R13_near | -9.5 | +3.2 |
| A1_likes_only | -12.3 | +3.6 |
| A2_dis_all | -9.4 | +5.0 |
| A3_dis_good | -8.5 | +6.4 |
| A4_mixed | -10.5 | +5.5 |
| A5_coral | -4.3 | -2.5 |
| A6_satisfaction | -8.9 | +3.3 |

## Standard metrics over all targets

Per user, averaged over the 300 users; the list = the 500 candidates in the reranker's order, a target outside the base list is a miss. Own persona:

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R10_mix | R11_mix_crit | R12_mix_reviews | R13_near | R13_near_ctrl | A1_likes_only | A2_dis_all | A3_dis_good | A4_mixed | A5_coral | A6_satisfaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@10 | 0.0203 | 0.0199 | 0.0213 | 0.0199 | 0.0213 | 0.0201 | 0.0185 | 0.0209 | 0.0203 | 0.0206 | 0.0233 | 0.0249 | 0.0206 | 0.0249 | 0.0219 | 0.0230 | 0.0228 | 0.0216 | 0.0227 |
| recall@20 | 0.0319 | 0.0326 | 0.0331 | 0.0321 | 0.0326 | 0.0327 | 0.0329 | 0.0316 | 0.0328 | 0.0342 | 0.0327 | 0.0338 | 0.0324 | 0.0366 | 0.0320 | 0.0350 | 0.0341 | 0.0335 | 0.0325 |
| recall@50 | 0.0638 | 0.0614 | 0.0604 | 0.0616 | 0.0604 | 0.0611 | 0.0615 | 0.0619 | 0.0623 | 0.0617 | 0.0624 | 0.0620 | 0.0629 | 0.0592 | 0.0637 | 0.0607 | 0.0663 | 0.0623 | 0.0617 |
| ndcg@10 | 0.0195 | 0.0195 | 0.0218 | 0.0195 | 0.0215 | 0.0195 | 0.0187 | 0.0198 | 0.0224 | 0.0225 | 0.0217 | 0.0219 | 0.0212 | 0.0230 | 0.0220 | 0.0229 | 0.0230 | 0.0220 | 0.0206 |
| ndcg@20 | 0.0235 | 0.0238 | 0.0251 | 0.0235 | 0.0247 | 0.0238 | 0.0236 | 0.0230 | 0.0261 | 0.0260 | 0.0240 | 0.0242 | 0.0243 | 0.0260 | 0.0248 | 0.0263 | 0.0259 | 0.0252 | 0.0236 |

the same with another user's persona:

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R10_mix | R11_mix_crit | R12_mix_reviews | R13_near | A1_likes_only | A2_dis_all | A3_dis_good | A4_mixed | A5_coral | A6_satisfaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@10 | 0.0203 | 0.0203 | 0.0213 | 0.0203 | 0.0209 | 0.0205 | 0.0184 | 0.0204 | 0.0168 | 0.0167 | 0.0165 | 0.0187 | 0.0183 | 0.0173 | 0.0164 | 0.0157 | 0.0216 | 0.0178 |
| recall@20 | 0.0319 | 0.0321 | 0.0317 | 0.0321 | 0.0313 | 0.0321 | 0.0318 | 0.0319 | 0.0293 | 0.0301 | 0.0303 | 0.0294 | 0.0301 | 0.0296 | 0.0302 | 0.0293 | 0.0333 | 0.0298 |
| recall@50 | 0.0638 | 0.0576 | 0.0638 | 0.0576 | 0.0604 | 0.0585 | 0.0610 | 0.0605 | 0.0574 | 0.0553 | 0.0595 | 0.0603 | 0.0567 | 0.0603 | 0.0555 | 0.0594 | 0.0616 | 0.0585 |
| ndcg@10 | 0.0195 | 0.0197 | 0.0210 | 0.0197 | 0.0200 | 0.0194 | 0.0186 | 0.0195 | 0.0180 | 0.0175 | 0.0181 | 0.0188 | 0.0208 | 0.0187 | 0.0179 | 0.0173 | 0.0228 | 0.0182 |
| ndcg@20 | 0.0235 | 0.0239 | 0.0244 | 0.0239 | 0.0235 | 0.0237 | 0.0233 | 0.0234 | 0.0221 | 0.0221 | 0.0226 | 0.0221 | 0.0238 | 0.0218 | 0.0221 | 0.0210 | 0.0260 | 0.0222 |

each reranker minus the base ranker, paired per user (difference x100, 95% bootstrap interval, two-sided sign-flip p):

| reranker | recall@10 | recall@20 | recall@50 | ndcg@10 | ndcg@20 |
|---|---|---|---|---|---|
| R1_like | -0.04 [-0.22, +0.11] p 0.750 | +0.07 [-0.11, +0.36] p 0.816 | -0.24 [-0.78, +0.21] p 0.392 | +0.01 [-0.14, +0.15] p 0.946 | +0.03 [-0.06, +0.14] p 0.600 |
| R2_like-dis | +0.10 [-0.11, +0.31] p 0.373 | +0.12 [-0.14, +0.48] p 0.494 | -0.34 [-0.81, +0.02] p 0.218 | +0.23 [+0.03, +0.47] p 0.035 | +0.16 [-0.01, +0.38] p 0.112 |
| R3_soft | -0.04 [-0.22, +0.11] p 0.750 | +0.02 [-0.19, +0.31] p 0.968 | -0.22 [-0.76, +0.23] p 0.429 | +0.01 [-0.14, +0.16] p 0.940 | +0.00 [-0.10, +0.12] p 0.987 |
| R4_rocchio | +0.10 [-0.12, +0.31] p 0.406 | +0.07 [-0.15, +0.36] p 0.721 | -0.34 [-0.88, +0.07] p 0.249 | +0.21 [-0.01, +0.46] p 0.092 | +0.12 [-0.05, +0.33] p 0.251 |
| R5_naive | -0.02 [-0.20, +0.16] p 0.873 | +0.08 [-0.13, +0.38] p 0.694 | -0.27 [-0.81, +0.18] p 0.329 | +0.00 [-0.12, +0.12] p 0.930 | +0.03 [-0.06, +0.15] p 0.563 |
| R8_rev_like | -0.18 [-0.62, +0.11] p 0.501 | +0.10 [-0.06, +0.38] p 0.744 | -0.23 [-0.71, +0.15] p 0.350 | -0.08 [-0.30, +0.11] p 0.476 | +0.02 [-0.12, +0.15] p 0.836 |
| R9_rev | +0.06 [-0.12, +0.25] p 0.528 | -0.03 [-0.12, +0.03] p 0.750 | -0.19 [-0.66, +0.17] p 0.453 | +0.03 [-0.14, +0.19] p 0.728 | -0.05 [-0.17, +0.04] p 0.398 |
| R10_mix | -0.00 [-0.46, +0.37] p 0.985 | +0.10 [-0.33, +0.56] p 0.678 | -0.15 [-1.10, +0.78] p 0.756 | +0.30 [-0.04, +0.66] p 0.096 | +0.27 [-0.01, +0.60] p 0.085 |
| R11_mix_crit | +0.03 [-0.45, +0.39] p 0.909 | +0.24 [-0.20, +0.71] p 0.307 | -0.22 [-1.22, +0.74] p 0.669 | +0.30 [-0.05, +0.68] p 0.093 | +0.26 [+0.01, +0.52] p 0.047 |
| R12_mix_reviews | +0.30 [-0.10, +0.72] p 0.178 | +0.09 [-0.21, +0.39] p 0.598 | -0.14 [-0.82, +0.48] p 0.669 | +0.22 [-0.08, +0.53] p 0.170 | +0.05 [-0.17, +0.25] p 0.651 |
| R13_near | +0.46 [+0.12, +0.85] p 0.011 | +0.20 [-0.15, +0.56] p 0.289 | -0.18 [-0.80, +0.39] p 0.558 | +0.25 [-0.04, +0.54] p 0.104 | +0.07 [-0.16, +0.30] p 0.563 |
| R13_near_ctrl | +0.03 [-0.50, +0.49] p 0.915 | +0.06 [-0.24, +0.36] p 0.734 | -0.09 [-0.90, +0.64] p 0.827 | +0.18 [-0.13, +0.48] p 0.275 | +0.09 [-0.10, +0.27] p 0.362 |
| A1_likes_only | +0.46 [-0.04, +0.96] p 0.074 | +0.47 [+0.00, +0.99] p 0.057 | -0.46 [-1.20, +0.23] p 0.216 | +0.35 [+0.01, +0.69] p 0.049 | +0.25 [-0.04, +0.54] p 0.096 |
| A2_dis_all | +0.16 [-0.40, +0.66] p 0.586 | +0.01 [-0.40, +0.37] p 0.950 | -0.01 [-0.75, +0.74] p 0.977 | +0.26 [-0.08, +0.60] p 0.160 | +0.13 [-0.15, +0.42] p 0.386 |
| A3_dis_good | +0.27 [-0.39, +0.88] p 0.437 | +0.31 [-0.21, +0.83] p 0.257 | -0.31 [-1.20, +0.52] p 0.483 | +0.34 [-0.08, +0.75] p 0.123 | +0.28 [-0.06, +0.64] p 0.131 |
| A4_mixed | +0.25 [-0.35, +0.81] p 0.428 | +0.22 [-0.17, +0.64] p 0.296 | +0.24 [-0.40, +0.94] p 0.482 | +0.35 [-0.09, +0.80] p 0.144 | +0.24 [-0.11, +0.64] p 0.220 |
| A5_coral | +0.13 [-0.11, +0.41] p 0.413 | +0.16 [-0.16, +0.53] p 0.395 | -0.16 [-0.79, +0.45] p 0.630 | +0.25 [+0.05, +0.47] p 0.013 | +0.17 [-0.00, +0.36] p 0.061 |
| A6_satisfaction | +0.24 [-0.15, +0.65] p 0.272 | +0.06 [-0.23, +0.35] p 0.688 | -0.21 [-0.78, +0.32] p 0.465 | +0.12 [-0.14, +0.38] p 0.386 | +0.01 [-0.17, +0.19] p 0.928 |

## Paired comparisons per user

a - b per user: mean, 95% bootstrap interval, two-sided sign-flip p. In-list rank change: negative = a moves the targets higher than b; ndcg and recall x100: positive = a is better.

| comparison | in_list_change | ndcg@10 | recall@20 |
|---|---|---|---|
| R1_like: own - another user's persona | -5.2 [-7.5, -3.1] p<0.001 | -0.02 [-0.19, +0.13] p 0.834 | +0.05 [-0.17, +0.35] p 0.840 |
| R2_like-dis: own - another user's persona | -2.4 [-5.1, +0.0] p 0.067 | +0.08 [-0.12, +0.28] p 0.437 | +0.14 [-0.13, +0.50] p 0.434 |
| R3_soft: own - another user's persona | -4.3 [-6.6, -2.4] p<0.001 | -0.02 [-0.19, +0.14] p 0.819 | +0.00 [-0.24, +0.30] p 1.000 |
| R4_rocchio: own - another user's persona | -2.8 [-5.0, -0.7] p 0.011 | +0.15 [-0.05, +0.39] p 0.188 | +0.13 [-0.08, +0.43] p 0.468 |
| R5_naive: own - another user's persona | -4.7 [-7.0, -2.3] p<0.001 | +0.01 [-0.13, +0.15] p 0.875 | +0.06 [-0.14, +0.34] p 0.817 |
| R8_rev_like: own - another user's persona | -0.9 [-1.8, -0.0] p 0.062 | +0.01 [-0.11, +0.13] p 0.888 | +0.11 [+0.00, +0.33] p 1.000 |
| R9_rev: own - another user's persona | -1.5 [-3.0, -0.0] p 0.051 | +0.03 [-0.10, +0.15] p 0.690 | -0.03 [-0.14, +0.05] p 1.000 |
| R10_mix: own - another user's persona | -15.6 [-22.5, -8.3] p<0.001 | +0.45 [+0.05, +0.86] p 0.024 | +0.35 [-0.05, +0.78] p 0.104 |
| R11_mix_crit: own - another user's persona | -15.4 [-22.0, -8.6] p<0.001 | +0.50 [+0.16, +0.90] p 0.007 | +0.41 [-0.05, +0.91] p 0.086 |
| R12_mix_reviews: own - another user's persona | -13.3 [-19.9, -6.6] p<0.001 | +0.36 [+0.04, +0.68] p 0.028 | +0.24 [-0.24, +0.71] p 0.341 |
| R13_near: own - another user's persona | -12.6 [-19.3, -5.9] p<0.001 | +0.32 [+0.02, +0.63] p 0.039 | +0.45 [-0.01, +0.92] p 0.061 |
| A1_likes_only: own - another user's persona | -13.4 [-20.8, -5.8] p<0.001 | +0.22 [-0.14, +0.55] p 0.246 | +0.65 [+0.12, +1.25] p 0.021 |
| A2_dis_all: own - another user's persona | -17.7 [-25.4, -10.3] p<0.001 | +0.33 [-0.07, +0.74] p 0.095 | +0.24 [-0.30, +0.78] p 0.400 |
| A3_dis_good: own - another user's persona | -18.7 [-28.6, -9.3] p<0.001 | +0.50 [+0.08, +0.91] p 0.015 | +0.48 [+0.04, +0.94] p 0.031 |
| A4_mixed: own - another user's persona | -19.5 [-29.0, -9.7] p<0.001 | +0.57 [+0.15, +1.02] p 0.006 | +0.48 [-0.09, +1.05] p 0.103 |
| A5_coral: own - another user's persona | -1.0 [-3.3, +1.6] p 0.432 | -0.08 [-0.22, +0.03] p 0.253 | +0.02 [-0.16, +0.25] p 0.905 |
| A6_satisfaction: own - another user's persona | -11.6 [-17.6, -5.5] p<0.001 | +0.24 [-0.03, +0.52] p 0.083 | +0.27 [-0.14, +0.70] p 0.230 |
| R13_near - R12_mix_reviews | +0.4 [-1.3, +2.2] p 0.651 | +0.03 [-0.10, +0.17] p 0.751 | +0.11 [-0.08, +0.34] p 0.324 |
| R13_near_ctrl - R12_mix_reviews | +0.8 [-0.9, +2.4] p 0.379 | -0.05 [-0.27, +0.17] p 0.695 | -0.03 [-0.11, +0.03] p 0.747 |
| R13_near - R13_near_ctrl | -0.4 [-2.5, +1.7] p 0.747 | +0.07 [-0.15, +0.32] p 0.569 | +0.14 [-0.02, +0.36] p 0.248 |
| A1_likes_only - R12_mix_reviews | -1.1 [-4.7, +2.5] p 0.533 | +0.13 [-0.05, +0.33] p 0.206 | +0.38 [-0.00, +0.83] p 0.053 |
| A2_dis_all - R12_mix_reviews | -1.4 [-4.5, +1.6] p 0.387 | +0.03 [-0.26, +0.37] p 0.839 | -0.07 [-0.35, +0.16] p 0.684 |
| A3_dis_good - R12_mix_reviews | -0.1 [-6.9, +6.5] p 0.984 | +0.12 [-0.25, +0.48] p 0.543 | +0.23 [-0.23, +0.71] p 0.368 |
| A4_mixed - R12_mix_reviews | -3.5 [-8.7, +1.6] p 0.187 | +0.13 [-0.20, +0.51] p 0.497 | +0.13 [-0.15, +0.44] p 0.403 |
| A5_coral - R12_mix_reviews | +5.9 [+0.4, +11.5] p 0.045 | +0.03 [-0.23, +0.30] p 0.814 | +0.07 [-0.26, +0.46] p 0.706 |
| A6_satisfaction - R12_mix_reviews | +1.9 [+0.0, +3.8] p 0.044 | -0.11 [-0.21, -0.02] p 0.020 | -0.02 [-0.11, +0.05] p 0.752 |

R10_mix weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.448 | 0.023 |
| like_s1 | +0.059 | 0.025 |
| like_s2 | +0.060 | 0.026 |
| like_s3 | +0.038 | 0.027 |
| like_s4 | +0.021 | 0.026 |
| like_s5 | +0.022 | 0.025 |
| dis_s1 | +0.028 | 0.025 |
| dis_s2 | +0.020 | 0.027 |
| dis_s3 | +0.026 | 0.027 |
| dis_s4 | +0.002 | 0.027 |
| dis_s5 | -0.033 | 0.026 |
| sum of the like weights | +0.200 | 0.036 |
| sum of the dislike weights | +0.043 | 0.036 |

R11_mix_crit weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.448 | 0.023 |
| like_s1 | +0.064 | 0.025 |
| like_s2 | +0.064 | 0.026 |
| like_s3 | +0.041 | 0.026 |
| like_s4 | +0.024 | 0.026 |
| like_s5 | +0.024 | 0.025 |
| dis_crit_s1 | +0.015 | 0.026 |
| dis_crit_s2 | +0.018 | 0.027 |
| dis_crit_s3 | +0.020 | 0.028 |
| dis_crit_s4 | +0.022 | 0.027 |
| dis_crit_s5 | +0.000 | 0.027 |
| no_crit | -0.015 | 0.022 |
| sum of the like weights | +0.217 | 0.033 |
| sum of the dislike weights | +0.074 | 0.038 |

R12_mix_reviews weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.444 | 0.023 |
| like_good_s1 | +0.045 | 0.024 |
| like_good_s2 | +0.043 | 0.026 |
| like_good_s3 | +0.045 | 0.026 |
| like_good_s4 | +0.043 | 0.026 |
| like_good_s5 | +0.000 | 0.025 |
| dis_bad_s1 | +0.034 | 0.025 |
| dis_bad_s2 | +0.037 | 0.027 |
| dis_bad_s3 | +0.015 | 0.027 |
| dis_bad_s4 | +0.015 | 0.027 |
| dis_bad_s5 | +0.025 | 0.026 |
| no_bad | -0.030 | 0.025 |
| n_good | -0.009 | 0.024 |
| n_bad | +0.048 | 0.024 |
| sum of the like weights | +0.176 | 0.034 |
| sum of the dislike weights | +0.127 | 0.037 |

A1_likes_only weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.443 | 0.023 |
| like_good_s1 | +0.072 | 0.038 |
| like_good_s2 | +0.051 | 0.047 |
| like_good_s3 | +0.066 | 0.050 |
| like_good_s4 | +0.086 | 0.048 |
| like_good_s5 | -0.038 | 0.040 |
| no_bad | -0.019 | 0.042 |
| n_good | -0.016 | 0.034 |
| n_bad | +0.128 | 0.037 |
| sum of the like weights | +0.236 | 0.039 |

A2_dis_all weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.441 | 0.023 |
| like_good_s1 | +0.040 | 0.024 |
| like_good_s2 | +0.037 | 0.026 |
| like_good_s3 | +0.040 | 0.026 |
| like_good_s4 | +0.038 | 0.026 |
| like_good_s5 | -0.005 | 0.025 |
| dis_all_s1 | +0.051 | 0.025 |
| dis_all_s2 | +0.043 | 0.026 |
| dis_all_s3 | +0.033 | 0.026 |
| dis_all_s4 | +0.044 | 0.026 |
| dis_all_s5 | +0.018 | 0.025 |
| no_bad | -0.013 | 0.025 |
| n_good | -0.012 | 0.024 |
| n_bad | +0.043 | 0.024 |
| no_rev | -0.006 | 0.030 |
| sum of the like weights | +0.151 | 0.035 |
| sum of the dislike weights | +0.189 | 0.036 |

A3_dis_good weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 1, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.434 | 0.023 |
| like_good_s1 | +0.058 | 0.043 |
| like_good_s2 | +0.030 | 0.057 |
| like_good_s3 | +0.055 | 0.062 |
| like_good_s4 | +0.094 | 0.058 |
| like_good_s5 | -0.070 | 0.044 |
| dis_good_s1 | +0.104 | 0.045 |
| dis_good_s2 | +0.042 | 0.061 |
| dis_good_s3 | +0.063 | 0.068 |
| dis_good_s4 | +0.029 | 0.065 |
| dis_good_s5 | +0.003 | 0.051 |
| no_bad | -0.007 | 0.048 |
| n_good | -0.055 | 0.038 |
| n_bad | +0.141 | 0.042 |
| no_good | -1.391 | 0.696 |
| sum of the like weights | +0.166 | 0.043 |
| sum of the dislike weights | +0.240 | 0.042 |

A4_mixed weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.434 | 0.023 |
| mixed_s1 | +0.020 | 0.040 |
| mixed_s2 | +0.106 | 0.050 |
| mixed_s3 | +0.018 | 0.056 |
| mixed_s4 | +0.111 | 0.057 |
| mixed_s5 | -0.061 | 0.060 |
| mixed_s6 | -0.007 | 0.062 |
| mixed_s7 | +0.142 | 0.061 |
| mixed_s8 | -0.001 | 0.058 |
| mixed_s9 | +0.093 | 0.053 |
| mixed_s10 | -0.105 | 0.041 |
| no_bad | +0.003 | 0.042 |
| n_good | -0.018 | 0.034 |
| n_bad | +0.073 | 0.038 |
| no_rev | -0.035 | 0.078 |
| sum of the mixed-persona weights | +0.315 | 0.043 |

A5_coral weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.455 | 0.023 |
| coral | +0.038 | 0.024 |
| no_bad | -0.013 | 0.042 |
| n_good | +0.014 | 0.034 |
| n_bad | +0.126 | 0.038 |

A6_satisfaction weights: logistic regression on 414 training users with a target inside the list (633 targets rated 4-5 stars, 207000 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.434 | 0.029 |
| like_good_s1 | +0.036 | 0.026 |
| like_good_s2 | +0.031 | 0.027 |
| like_good_s3 | +0.045 | 0.027 |
| like_good_s4 | +0.038 | 0.027 |
| like_good_s5 | +0.005 | 0.026 |
| dis_bad_s1 | +0.029 | 0.027 |
| dis_bad_s2 | +0.026 | 0.028 |
| dis_bad_s3 | +0.005 | 0.028 |
| dis_bad_s4 | +0.013 | 0.028 |
| dis_bad_s5 | +0.012 | 0.028 |
| no_bad | -0.023 | 0.026 |
| n_good | -0.007 | 0.026 |
| n_bad | +0.020 | 0.026 |
| sum of the like weights | +0.155 | 0.041 |
| sum of the dislike weights | +0.085 | 0.043 |

R13_near: R12_mix_reviews's inputs and L2 penalty (1000), refit per test user with the 414 training users weighted by exp(alpha * z(persona closeness)). alpha picked by leave-one-user-out over the training users (largest held-out log-likelihood): **1**

| alpha | held-out log-likelihood | held-out in-list rank change | effective training users |
|---|---|---|---|
| 0 | -6236.06 | -8.83 | 413 |
| 0.25 | -6234.89 | -8.85 | 392 |
| 0.5 | -6233.98 | -8.87 | 340 |
| 1 | -6233.58 | -9.02 | 212 |
| 2 | -6250.95 | -7.62 | 64 |

R13_near weights over the test users (own persona): mean, sd, min, max

| weight | mean | sd | min | max |
|---|---|---|---|---|
| base | +0.454 | 0.033 | +0.309 | +0.535 |
| like_sum | +0.170 | 0.027 | +0.083 | +0.243 |
| dis_sum | +0.110 | 0.032 | +0.010 | +0.202 |

R13_near weights over the test users (closeness from another user's persona): mean, sd, min, max

| weight | mean | sd | min | max |
|---|---|---|---|---|
| base | +0.454 | 0.033 | +0.309 | +0.535 |
| like_sum | +0.170 | 0.027 | +0.083 | +0.243 |
| dis_sum | +0.110 | 0.032 | +0.010 | +0.202 |
