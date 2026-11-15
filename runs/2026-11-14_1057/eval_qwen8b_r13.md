run 2026-11-14_1057 | encoder Qwen/Qwen3-Embedding-8B | alpha 0.1 beta 0.1 gamma 1.0

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R10_mix | R11_mix_crit | R12_mix_reviews | R13_near | R13_near_ctrl | A1_likes_only | A2_dis_all | A3_dis_good | A4_mixed | A5_coral | A6_satisfaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 192.5 | 195.4 | 193.3 | 193.1 | 190.8 | 194.0 | 195.1 | 183.9 | 183.9 | 177.2 | 178.8 | 177.7 | 179.8 | 178.1 | 183.1 | 179.5 | 195.7 | 180.4 |
| in_list_rank_change | 0.0 | -3.6 | -0.7 | -2.8 | -3.0 | -5.3 | -2.0 | -0.9 | -12.1 | -12.2 | -18.9 | -17.3 | -18.4 | -16.3 | -18.0 | -12.9 | -16.6 | -0.3 | -15.7 |
| in_list_moved_up | 0 | 265 | 238 | 261 | 262 | 294 | 330 | 198 | 277 | 291 | 285 | 285 | 288 | 284 | 290 | 291 | 292 | 206 | 280 |
| in_list_stayed | 514 | 49 | 39 | 46 | 43 | 38 | 57 | 49 | 8 | 13 | 13 | 6 | 4 | 5 | 6 | 11 | 7 | 124 | 10 |
| in_list_moved_down | 0 | 200 | 237 | 207 | 209 | 182 | 127 | 267 | 229 | 210 | 216 | 223 | 222 | 225 | 218 | 212 | 215 | 184 | 224 |
| in_list_top10 | 27 | 29 | 26 | 29 | 26 | 28 | 27 | 27 | 31 | 31 | 31 | 32 | 31 | 32 | 30 | 30 | 31 | 27 | 31 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 32 | 41 | 30 | 39 | 36 | 47 | 37 | 384 | 286 | 621 | 634 | 629 | 498 | 697 | 556 | 632 | 6 | 603 |
| all_mean_rank | 438.8 | 437.8 | 438.4 | 438.0 | 437.9 | 437.4 | 438.1 | 438.4 | 429.8 | 433.1 | 423.4 | 423.4 | 423.4 | 426.1 | 421.6 | 427.5 | 423.5 | 438.7 | 424.1 |
| in_list_change_liked | 0.0 | -5.0 | -1.9 | -4.1 | -4.3 | -6.5 | -2.6 | -1.1 | -17.0 | -16.9 | -25.5 | -23.9 | -25.8 | -24.6 | -25.2 | -17.9 | -22.0 | -0.9 | -23.1 |
| in_list_change_disliked | 0.0 | 0.1 | 2.8 | 0.8 | 0.1 | -2.1 | 0.1 | 0.6 | -1.8 | 0.3 | -8.2 | -5.3 | -4.7 | 0.7 | -7.3 | -2.6 | -6.6 | 1.4 | -1.2 |

mean rank change of the in-list targets per user, bootstrap over users (95% interval):

| reranker | mean change | 2.5% | 97.5% |
|---|---|---|---|
| R1_like | -4.0 | -6.4 | -1.6 |
| R2_like-dis | -0.3 | -3.2 | +2.7 |
| R3_soft | -3.0 | -5.6 | -0.5 |
| R4_rocchio | -2.6 | -5.4 | +0.1 |
| R5_naive | -5.6 | -7.9 | -3.3 |
| R8_rev_like | -1.5 | -3.8 | +0.9 |
| R9_rev | -0.0 | -2.0 | +1.9 |
| R10_mix | -16.8 | -26.5 | -6.9 |
| R11_mix_crit | -15.9 | -23.9 | -7.7 |
| R12_mix_reviews | -17.3 | -27.8 | -6.4 |
| R13_near | -15.9 | -26.3 | -4.9 |
| R13_near_ctrl | -16.3 | -27.0 | -4.8 |
| A1_likes_only | -13.3 | -23.1 | -3.1 |
| A2_dis_all | -19.2 | -29.9 | -8.3 |
| A3_dis_good | -13.1 | -22.9 | -2.6 |
| A4_mixed | -17.9 | -29.2 | -6.0 |
| A5_coral | -0.2 | -0.8 | +0.4 |
| A6_satisfaction | -12.4 | -22.7 | -1.6 |

control: the same rerankers with another user's persona (what any persona moves):

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R10_mix | R11_mix_crit | R12_mix_reviews | R13_near | A1_likes_only | A2_dis_all | A3_dis_good | A4_mixed | A5_coral | A6_satisfaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 195.1 | 194.6 | 195.3 | 194.7 | 195.2 | 195.4 | 196.2 | 201.6 | 196.0 | 202.4 | 201.4 | 199.4 | 204.0 | 202.4 | 203.2 | 195.8 | 202.7 |
| in_list_rank_change | 0.0 | -1.0 | -1.5 | -0.8 | -1.3 | -0.9 | -0.6 | 0.1 | 5.6 | -0.0 | 6.3 | 5.4 | 3.4 | 8.0 | 6.4 | 7.1 | -0.3 | 6.6 |
| in_list_moved_up | 0 | 232 | 255 | 230 | 244 | 241 | 323 | 189 | 221 | 237 | 236 | 240 | 240 | 225 | 227 | 238 | 222 | 229 |
| in_list_stayed | 514 | 43 | 49 | 45 | 43 | 36 | 65 | 51 | 6 | 11 | 6 | 7 | 6 | 8 | 9 | 12 | 109 | 6 |
| in_list_moved_down | 0 | 239 | 210 | 239 | 227 | 237 | 126 | 274 | 287 | 266 | 272 | 267 | 268 | 281 | 278 | 264 | 183 | 279 |
| in_list_top10 | 27 | 26 | 26 | 26 | 26 | 27 | 28 | 27 | 28 | 29 | 24 | 23 | 27 | 24 | 28 | 23 | 27 | 25 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 32 | 42 | 32 | 42 | 40 | 50 | 32 | 371 | 271 | 595 | 609 | 490 | 650 | 546 | 613 | 8 | 573 |
| all_mean_rank | 438.8 | 438.4 | 438.3 | 438.5 | 438.3 | 438.4 | 438.4 | 438.6 | 434.4 | 436.0 | 429.6 | 428.8 | 431.4 | 428.6 | 431.5 | 429.2 | 438.7 | 430.0 |
| in_list_change_liked | 0.0 | -1.7 | -2.5 | -1.8 | -2.5 | -1.2 | -1.1 | 0.1 | 5.4 | -1.2 | 5.7 | 3.8 | 1.2 | 8.1 | 6.7 | 7.5 | -0.6 | 3.6 |
| in_list_change_disliked | 0.0 | 0.3 | 1.4 | 1.4 | 1.1 | -0.9 | 1.1 | 0.8 | 2.1 | 2.5 | 5.1 | 7.4 | 1.9 | 3.6 | -0.0 | 2.8 | -0.0 | 10.0 |

| reranker | in-list rank change, own persona | another user's persona |
|---|---|---|
| R1_like | -3.6 | -1.0 |
| R2_like-dis | -0.7 | -1.5 |
| R3_soft | -2.8 | -0.8 |
| R4_rocchio | -3.0 | -1.3 |
| R5_naive | -5.3 | -0.9 |
| R8_rev_like | -2.0 | -0.6 |
| R9_rev | -0.9 | +0.1 |
| R10_mix | -12.1 | +5.6 |
| R11_mix_crit | -12.2 | -0.0 |
| R12_mix_reviews | -18.9 | +6.3 |
| R13_near | -17.3 | +5.4 |
| A1_likes_only | -16.3 | +3.4 |
| A2_dis_all | -18.0 | +8.0 |
| A3_dis_good | -12.9 | +6.4 |
| A4_mixed | -16.6 | +7.1 |
| A5_coral | -0.3 | -0.3 |
| A6_satisfaction | -15.7 | +6.6 |

## Standard metrics over all targets

Per user, averaged over the 300 users; the list = the 500 candidates in the reranker's order, a target outside the base list is a miss. Own persona:

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R10_mix | R11_mix_crit | R12_mix_reviews | R13_near | R13_near_ctrl | A1_likes_only | A2_dis_all | A3_dis_good | A4_mixed | A5_coral | A6_satisfaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@10 | 0.0120 | 0.0124 | 0.0119 | 0.0124 | 0.0119 | 0.0122 | 0.0120 | 0.0120 | 0.0142 | 0.0153 | 0.0142 | 0.0149 | 0.0140 | 0.0150 | 0.0131 | 0.0126 | 0.0133 | 0.0120 | 0.0140 |
| recall@20 | 0.0216 | 0.0210 | 0.0219 | 0.0219 | 0.0211 | 0.0214 | 0.0211 | 0.0210 | 0.0227 | 0.0234 | 0.0205 | 0.0190 | 0.0206 | 0.0211 | 0.0220 | 0.0221 | 0.0215 | 0.0210 | 0.0205 |
| recall@50 | 0.0427 | 0.0444 | 0.0425 | 0.0440 | 0.0430 | 0.0443 | 0.0432 | 0.0428 | 0.0462 | 0.0438 | 0.0463 | 0.0444 | 0.0475 | 0.0465 | 0.0466 | 0.0494 | 0.0414 | 0.0431 | 0.0439 |
| ndcg@10 | 0.0121 | 0.0129 | 0.0121 | 0.0129 | 0.0119 | 0.0123 | 0.0117 | 0.0119 | 0.0138 | 0.0141 | 0.0144 | 0.0149 | 0.0147 | 0.0135 | 0.0134 | 0.0126 | 0.0134 | 0.0121 | 0.0137 |
| ndcg@20 | 0.0149 | 0.0150 | 0.0150 | 0.0152 | 0.0148 | 0.0147 | 0.0143 | 0.0144 | 0.0160 | 0.0159 | 0.0159 | 0.0154 | 0.0164 | 0.0148 | 0.0160 | 0.0153 | 0.0157 | 0.0147 | 0.0155 |

the same with another user's persona:

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R10_mix | R11_mix_crit | R12_mix_reviews | R13_near | A1_likes_only | A2_dis_all | A3_dis_good | A4_mixed | A5_coral | A6_satisfaction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recall@10 | 0.0120 | 0.0109 | 0.0109 | 0.0109 | 0.0119 | 0.0111 | 0.0122 | 0.0120 | 0.0126 | 0.0133 | 0.0096 | 0.0094 | 0.0113 | 0.0090 | 0.0123 | 0.0110 | 0.0120 | 0.0116 |
| recall@20 | 0.0216 | 0.0210 | 0.0199 | 0.0214 | 0.0207 | 0.0211 | 0.0215 | 0.0211 | 0.0214 | 0.0215 | 0.0178 | 0.0163 | 0.0204 | 0.0165 | 0.0205 | 0.0181 | 0.0211 | 0.0183 |
| recall@50 | 0.0427 | 0.0434 | 0.0433 | 0.0430 | 0.0433 | 0.0428 | 0.0437 | 0.0433 | 0.0388 | 0.0475 | 0.0422 | 0.0401 | 0.0467 | 0.0382 | 0.0445 | 0.0428 | 0.0427 | 0.0401 |
| ndcg@10 | 0.0121 | 0.0114 | 0.0112 | 0.0115 | 0.0115 | 0.0114 | 0.0119 | 0.0122 | 0.0110 | 0.0119 | 0.0101 | 0.0096 | 0.0109 | 0.0100 | 0.0113 | 0.0104 | 0.0122 | 0.0106 |
| ndcg@20 | 0.0149 | 0.0145 | 0.0138 | 0.0147 | 0.0142 | 0.0144 | 0.0145 | 0.0148 | 0.0138 | 0.0143 | 0.0121 | 0.0113 | 0.0134 | 0.0116 | 0.0134 | 0.0124 | 0.0147 | 0.0124 |

each reranker minus the base ranker, paired per user (difference x100, 95% bootstrap interval, two-sided sign-flip p):

| reranker | recall@10 | recall@20 | recall@50 | ndcg@10 | ndcg@20 |
|---|---|---|---|---|---|
| R1_like | +0.03 [+0.00, +0.08] p 0.498 | -0.06 [-0.24, +0.07] p 0.626 | +0.17 [-0.07, +0.54] p 0.500 | +0.07 [-0.03, +0.20] p 0.215 | +0.00 [-0.11, +0.12] p 0.964 |
| R2_like-dis | -0.02 [-0.05, +0.00] p 1.000 | +0.03 [-0.21, +0.32] p 0.876 | -0.03 [-0.20, +0.13] p 0.803 | -0.00 [-0.11, +0.11] p 0.996 | +0.01 [-0.12, +0.15] p 0.875 |
| R3_soft | +0.03 [+0.00, +0.08] p 0.511 | +0.03 [-0.21, +0.32] p 0.876 | +0.13 [-0.17, +0.52] p 0.666 | +0.07 [-0.03, +0.21] p 0.201 | +0.03 [-0.10, +0.17] p 0.717 |
| R4_rocchio | -0.02 [-0.05, +0.00] p 1.000 | -0.05 [-0.20, +0.05] p 1.000 | +0.02 [-0.09, +0.12] p 0.693 | -0.02 [-0.10, +0.05] p 0.711 | -0.02 [-0.11, +0.06] p 0.874 |
| R5_naive | +0.02 [-0.03, +0.08] p 1.000 | -0.03 [-0.08, +0.00] p 1.000 | +0.15 [-0.08, +0.53] p 0.658 | +0.02 [-0.08, +0.11] p 0.679 | -0.02 [-0.09, +0.03] p 0.598 |
| R8_rev_like | -0.00 [-0.08, +0.07] p 1.000 | -0.05 [-0.20, +0.05] p 1.000 | +0.05 [+0.00, +0.13] p 0.505 | -0.04 [-0.15, +0.04] p 0.479 | -0.06 [-0.17, +0.02] p 0.362 |
| R9_rev | +0.00 [+0.00, +0.00] p 1.000 | -0.07 [-0.23, +0.03] p 0.757 | +0.00 [-0.09, +0.08] p 1.000 | -0.02 [-0.09, +0.02] p 0.631 | -0.05 [-0.14, +0.02] p 0.295 |
| R10_mix | +0.21 [-0.15, +0.63] p 0.321 | +0.10 [-0.09, +0.33] p 0.378 | +0.35 [-0.15, +0.94] p 0.257 | +0.17 [-0.12, +0.51] p 0.320 | +0.11 [-0.11, +0.38] p 0.477 |
| R11_mix_crit | +0.32 [-0.03, +0.76] p 0.154 | +0.18 [-0.09, +0.51] p 0.277 | +0.11 [-0.11, +0.37] p 0.406 | +0.20 [-0.01, +0.43] p 0.076 | +0.10 [-0.07, +0.28] p 0.261 |
| R12_mix_reviews | +0.22 [-0.25, +0.69] p 0.382 | -0.12 [-0.60, +0.33] p 0.632 | +0.36 [-0.31, +0.98] p 0.299 | +0.23 [-0.17, +0.65] p 0.264 | +0.10 [-0.23, +0.47] p 0.567 |
| R13_near | +0.29 [-0.15, +0.73] p 0.205 | -0.27 [-0.73, +0.16] p 0.251 | +0.16 [-0.56, +0.86] p 0.678 | +0.27 [-0.10, +0.68] p 0.178 | +0.05 [-0.28, +0.42] p 0.793 |
| R13_near_ctrl | +0.20 [-0.29, +0.67] p 0.423 | -0.10 [-0.61, +0.37] p 0.702 | +0.48 [-0.37, +1.32] p 0.284 | +0.25 [-0.17, +0.70] p 0.250 | +0.14 [-0.19, +0.51] p 0.438 |
| A1_likes_only | +0.30 [-0.13, +0.76] p 0.200 | -0.05 [-0.58, +0.45] p 0.856 | +0.37 [-0.46, +1.16] p 0.390 | +0.14 [-0.15, +0.43] p 0.339 | -0.02 [-0.28, +0.24] p 0.912 |
| A2_dis_all | +0.11 [-0.32, +0.52] p 0.629 | +0.03 [-0.52, +0.58] p 0.910 | +0.38 [-0.25, +0.97] p 0.247 | +0.12 [-0.26, +0.51] p 0.534 | +0.11 [-0.25, +0.48] p 0.562 |
| A3_dis_good | +0.06 [-0.37, +0.50] p 0.801 | +0.05 [-0.50, +0.57] p 0.858 | +0.66 [-0.10, +1.45] p 0.098 | +0.05 [-0.29, +0.36] p 0.765 | +0.04 [-0.30, +0.35] p 0.821 |
| A4_mixed | +0.13 [-0.32, +0.62] p 0.604 | -0.02 [-0.56, +0.48] p 0.953 | -0.13 [-0.96, +0.64] p 0.759 | +0.13 [-0.24, +0.52] p 0.511 | +0.08 [-0.26, +0.44] p 0.659 |
| A5_coral | +0.00 [+0.00, +0.00] p 1.000 | -0.07 [-0.20, +0.00] p 1.000 | +0.03 [+0.00, +0.08] p 0.493 | +0.00 [+0.00, +0.00] p 1.000 | -0.02 [-0.08, +0.00] p 0.939 |
| A6_satisfaction | +0.20 [-0.28, +0.66] p 0.415 | -0.11 [-0.60, +0.32] p 0.625 | +0.12 [-0.65, +0.83] p 0.755 | +0.15 [-0.22, +0.54] p 0.418 | +0.06 [-0.26, +0.39] p 0.741 |

## Paired comparisons per user

a - b per user: mean, 95% bootstrap interval, two-sided sign-flip p. In-list rank change: negative = a moves the targets higher than b; ndcg and recall x100: positive = a is better.

| comparison | in_list_change | ndcg@10 | recall@20 |
|---|---|---|---|
| R1_like: own - another user's persona | -2.6 [-5.4, +0.4] p 0.087 | +0.14 [+0.03, +0.30] p 0.028 | +0.00 [-0.05, +0.05] p 1.000 |
| R2_like-dis: own - another user's persona | +0.5 [-3.0, +4.2] p 0.788 | +0.09 [-0.03, +0.25] p 0.238 | +0.21 [-0.01, +0.50] p 0.186 |
| R3_soft: own - another user's persona | -2.2 [-5.0, +0.8] p 0.165 | +0.14 [+0.02, +0.30] p 0.049 | +0.06 [-0.12, +0.31] p 0.933 |
| R4_rocchio: own - another user's persona | -1.5 [-4.6, +1.6] p 0.363 | +0.05 [-0.01, +0.12] p 0.242 | +0.05 [+0.00, +0.12] p 0.496 |
| R5_naive: own - another user's persona | -4.8 [-7.6, -2.0] p<0.001 | +0.09 [-0.04, +0.23] p 0.211 | +0.03 [-0.10, +0.20] p 0.814 |
| R8_rev_like: own - another user's persona | -0.8 [-1.9, +0.3] p 0.181 | -0.02 [-0.08, +0.04] p 1.000 | -0.03 [-0.12, +0.03] p 0.748 |
| R9_rev: own - another user's persona | -0.5 [-2.1, +1.1] p 0.570 | -0.03 [-0.09, +0.01] p 0.502 | -0.02 [-0.10, +0.05] p 1.000 |
| R10_mix: own - another user's persona | -15.6 [-26.8, -4.3] p 0.006 | +0.29 [+0.01, +0.61] p 0.064 | +0.13 [-0.44, +0.69] p 0.680 |
| R11_mix_crit: own - another user's persona | -13.3 [-21.4, -5.5] p<0.001 | +0.23 [-0.00, +0.48] p 0.074 | +0.20 [-0.35, +0.75] p 0.514 |
| R12_mix_reviews: own - another user's persona | -22.7 [-34.8, -10.3] p<0.001 | +0.44 [+0.01, +0.89] p 0.047 | +0.27 [-0.23, +0.75] p 0.302 |
| R13_near: own - another user's persona | -20.7 [-33.4, -8.6] p<0.001 | +0.53 [+0.11, +1.00] p 0.018 | +0.27 [-0.24, +0.77] p 0.301 |
| A1_likes_only: own - another user's persona | -15.6 [-27.3, -3.2] p 0.014 | +0.27 [-0.08, +0.60] p 0.131 | +0.08 [-0.44, +0.55] p 0.773 |
| A2_dis_all: own - another user's persona | -27.7 [-39.4, -15.9] p<0.001 | +0.34 [-0.08, +0.80] p 0.140 | +0.54 [-0.06, +1.13] p 0.082 |
| A3_dis_good: own - another user's persona | -19.7 [-30.6, -8.7] p<0.001 | +0.14 [-0.20, +0.46] p 0.416 | +0.17 [-0.51, +0.80] p 0.631 |
| A4_mixed: own - another user's persona | -23.8 [-36.1, -10.7] p<0.001 | +0.31 [-0.07, +0.73] p 0.142 | +0.33 [-0.21, +0.89] p 0.244 |
| A5_coral: own - another user's persona | +0.1 [-0.6, +0.8] p 0.765 | -0.00 [-0.00, +0.00] p 1.000 | -0.02 [-0.10, +0.05] p 1.000 |
| A6_satisfaction: own - another user's persona | -18.5 [-30.5, -6.4] p 0.002 | +0.31 [-0.07, +0.74] p 0.139 | +0.22 [-0.31, +0.75] p 0.425 |
| R13_near - R12_mix_reviews | +1.5 [-0.6, +3.6] p 0.189 | +0.04 [-0.09, +0.19] p 0.540 | -0.15 [-0.36, -0.00] p 0.098 |
| R13_near_ctrl - R12_mix_reviews | +1.0 [-1.0, +3.1] p 0.354 | +0.02 [-0.15, +0.19] p 0.803 | +0.02 [-0.22, +0.23] p 1.000 |
| R13_near - R13_near_ctrl | +0.5 [-2.1, +3.0] p 0.727 | +0.02 [-0.12, +0.18] p 0.772 | -0.16 [-0.35, -0.03] p 0.049 |
| A1_likes_only - R12_mix_reviews | +4.0 [-0.9, +8.8] p 0.114 | -0.09 [-0.37, +0.13] p 0.491 | +0.07 [-0.16, +0.36] p 0.728 |
| A2_dis_all - R12_mix_reviews | -1.8 [-5.5, +1.8] p 0.327 | -0.11 [-0.24, -0.00] p 0.090 | +0.15 [-0.07, +0.45] p 0.325 |
| A3_dis_good - R12_mix_reviews | +4.2 [-3.7, +12.4] p 0.299 | -0.18 [-0.44, +0.06] p 0.165 | +0.17 [-0.31, +0.67] p 0.516 |
| A4_mixed - R12_mix_reviews | -0.6 [-6.2, +5.0] p 0.822 | -0.10 [-0.34, +0.12] p 0.391 | +0.10 [-0.23, +0.47] p 0.580 |
| A5_coral - R12_mix_reviews | +17.1 [+6.2, +27.7] p 0.002 | -0.23 [-0.65, +0.17] p 0.264 | +0.05 [-0.38, +0.52] p 0.830 |
| A6_satisfaction - R12_mix_reviews | +5.0 [+1.6, +8.2] p 0.003 | -0.08 [-0.26, +0.09] p 0.397 | +0.01 [-0.20, +0.16] p 1.000 |

R10_mix weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.301 | 0.025 |
| like_s1 | +0.070 | 0.042 |
| like_s2 | +0.060 | 0.051 |
| like_s3 | +0.044 | 0.054 |
| like_s4 | +0.001 | 0.052 |
| like_s5 | -0.021 | 0.044 |
| dis_s1 | +0.113 | 0.044 |
| dis_s2 | -0.069 | 0.056 |
| dis_s3 | +0.068 | 0.059 |
| dis_s4 | -0.004 | 0.057 |
| dis_s5 | -0.096 | 0.047 |
| sum of the like weights | +0.155 | 0.045 |
| sum of the dislike weights | +0.012 | 0.045 |

R11_mix_crit weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.305 | 0.025 |
| like_s1 | +0.050 | 0.025 |
| like_s2 | +0.042 | 0.026 |
| like_s3 | +0.030 | 0.026 |
| like_s4 | +0.011 | 0.026 |
| like_s5 | -0.004 | 0.025 |
| dis_crit_s1 | +0.006 | 0.026 |
| dis_crit_s2 | +0.013 | 0.027 |
| dis_crit_s3 | +0.031 | 0.028 |
| dis_crit_s4 | +0.013 | 0.027 |
| dis_crit_s5 | -0.008 | 0.027 |
| no_crit | -0.028 | 0.023 |
| sum of the like weights | +0.129 | 0.034 |
| sum of the dislike weights | +0.055 | 0.039 |

R12_mix_reviews weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.305 | 0.025 |
| like_good_s1 | +0.118 | 0.039 |
| like_good_s2 | -0.013 | 0.048 |
| like_good_s3 | +0.065 | 0.051 |
| like_good_s4 | +0.103 | 0.049 |
| like_good_s5 | -0.056 | 0.040 |
| dis_bad_s1 | +0.106 | 0.042 |
| dis_bad_s2 | +0.027 | 0.052 |
| dis_bad_s3 | -0.022 | 0.057 |
| dis_bad_s4 | +0.015 | 0.056 |
| dis_bad_s5 | +0.010 | 0.048 |
| no_bad | +0.012 | 0.037 |
| n_good | -0.034 | 0.034 |
| n_bad | -0.011 | 0.039 |
| sum of the like weights | +0.218 | 0.041 |
| sum of the dislike weights | +0.135 | 0.044 |

A1_likes_only weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.309 | 0.025 |
| like_good_s1 | +0.123 | 0.039 |
| like_good_s2 | -0.010 | 0.048 |
| like_good_s3 | +0.067 | 0.051 |
| like_good_s4 | +0.106 | 0.049 |
| like_good_s5 | -0.054 | 0.040 |
| no_bad | +0.025 | 0.037 |
| n_good | -0.039 | 0.034 |
| n_bad | +0.019 | 0.036 |
| sum of the like weights | +0.232 | 0.041 |

A2_dis_all weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.304 | 0.025 |
| like_good_s1 | +0.111 | 0.039 |
| like_good_s2 | -0.018 | 0.048 |
| like_good_s3 | +0.061 | 0.051 |
| like_good_s4 | +0.099 | 0.049 |
| like_good_s5 | -0.060 | 0.040 |
| dis_all_s1 | +0.121 | 0.039 |
| dis_all_s2 | -0.011 | 0.049 |
| dis_all_s3 | +0.030 | 0.052 |
| dis_all_s4 | +0.030 | 0.051 |
| dis_all_s5 | +0.017 | 0.043 |
| no_bad | +0.039 | 0.037 |
| n_good | -0.042 | 0.034 |
| n_bad | -0.010 | 0.037 |
| no_rev | -0.007 | 0.097 |
| sum of the like weights | +0.193 | 0.042 |
| sum of the dislike weights | +0.187 | 0.044 |

A3_dis_good weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.308 | 0.025 |
| like_good_s1 | +0.054 | 0.025 |
| like_good_s2 | +0.018 | 0.026 |
| like_good_s3 | +0.035 | 0.026 |
| like_good_s4 | +0.040 | 0.026 |
| like_good_s5 | -0.012 | 0.025 |
| dis_good_s1 | +0.056 | 0.025 |
| dis_good_s2 | +0.035 | 0.026 |
| dis_good_s3 | +0.030 | 0.027 |
| dis_good_s4 | +0.031 | 0.026 |
| dis_good_s5 | +0.022 | 0.025 |
| no_bad | +0.007 | 0.024 |
| n_good | -0.026 | 0.024 |
| n_bad | +0.007 | 0.024 |
| no_good | +0.007 | 0.030 |
| sum of the like weights | +0.135 | 0.036 |
| sum of the dislike weights | +0.175 | 0.036 |

A4_mixed weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.304 | 0.025 |
| mixed_s1 | +0.088 | 0.040 |
| mixed_s2 | +0.073 | 0.050 |
| mixed_s3 | +0.002 | 0.055 |
| mixed_s4 | +0.063 | 0.058 |
| mixed_s5 | -0.018 | 0.060 |
| mixed_s6 | -0.028 | 0.061 |
| mixed_s7 | +0.056 | 0.061 |
| mixed_s8 | +0.033 | 0.058 |
| mixed_s9 | +0.095 | 0.052 |
| mixed_s10 | -0.087 | 0.041 |
| no_bad | +0.038 | 0.037 |
| n_good | -0.035 | 0.034 |
| n_bad | -0.021 | 0.037 |
| no_rev | -0.007 | 0.097 |
| sum of the mixed-persona weights | +0.279 | 0.044 |

A5_coral weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 10000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.315 | 0.025 |
| coral | +0.005 | 0.009 |
| no_bad | +0.002 | 0.010 |
| n_good | -0.001 | 0.010 |
| n_bad | -0.000 | 0.010 |

A6_satisfaction weights: logistic regression on 413 training users with a target inside the list (609 targets rated 4-5 stars, 206500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.328 | 0.031 |
| like_good_s1 | +0.123 | 0.046 |
| like_good_s2 | -0.068 | 0.056 |
| like_good_s3 | +0.092 | 0.059 |
| like_good_s4 | +0.117 | 0.056 |
| like_good_s5 | -0.024 | 0.047 |
| dis_bad_s1 | +0.092 | 0.050 |
| dis_bad_s2 | +0.061 | 0.059 |
| dis_bad_s3 | -0.063 | 0.064 |
| dis_bad_s4 | +0.051 | 0.063 |
| dis_bad_s5 | -0.039 | 0.055 |
| no_bad | +0.017 | 0.043 |
| n_good | -0.040 | 0.041 |
| n_bad | -0.042 | 0.046 |
| sum of the like weights | +0.239 | 0.050 |
| sum of the dislike weights | +0.102 | 0.053 |

R13_near: R12_mix_reviews's inputs and L2 penalty (100), refit per test user with the 413 training users weighted by exp(alpha * z(persona closeness)). alpha picked by leave-one-user-out over the training users (largest held-out log-likelihood): **0.5**

| alpha | held-out log-likelihood | held-out in-list rank change | effective training users |
|---|---|---|---|
| 0 | -5943.14 | -4.94 | 412 |
| 0.25 | -5942.20 | -4.76 | 389 |
| 0.5 | -5941.51 | -4.53 | 334 |
| 1 | -5944.74 | -3.39 | 199 |
| 2 | -6022.34 | +3.27 | 55 |

R13_near weights over the test users (own persona): mean, sd, min, max

| weight | mean | sd | min | max |
|---|---|---|---|---|
| base | +0.293 | 0.009 | +0.262 | +0.327 |
| like_sum | +0.216 | 0.018 | +0.164 | +0.277 |
| dis_sum | +0.132 | 0.015 | +0.088 | +0.174 |

R13_near weights over the test users (closeness from another user's persona): mean, sd, min, max

| weight | mean | sd | min | max |
|---|---|---|---|---|
| base | +0.293 | 0.009 | +0.262 | +0.327 |
| like_sum | +0.216 | 0.018 | +0.164 | +0.277 |
| dis_sum | +0.132 | 0.015 | +0.088 | +0.174 |
