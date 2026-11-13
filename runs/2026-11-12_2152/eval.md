| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm |
|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 199.5 | 204.8 | 200.3 | 207.9 | 203.1 | 201.8 | 203.7 | 217.1 |
| in_list_rank_change | 0.0 | 3.5 | 8.7 | 4.3 | 11.8 | 7.0 | 5.7 | 7.7 | 21.1 |
| in_list_moved_up | 0 | 235 | 234 | 230 | 234 | 227 | 352 | 178 | 168 |
| in_list_stayed | 514 | 5 | 7 | 8 | 5 | 9 | 13 | 7 | 4 |
| in_list_moved_down | 0 | 274 | 273 | 276 | 275 | 278 | 149 | 329 | 342 |
| in_list_top10 | 27 | 22 | 30 | 23 | 30 | 26 | 25 | 25 | 23 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 805 | 873 | 794 | 921 | 888 | 735 | 770 | 504 |
| all_mean_rank | 438.8 | 423.1 | 424.5 | 423.9 | 425.9 | 423.2 | 425.1 | 426.3 | 426.2 |

mean rank change of the in-list targets per user, bootstrap over users (95% interval):

| reranker | mean change | 2.5% | 97.5% |
|---|---|---|---|
| R1_like | -1.3 | -14.9 | +11.8 |
| R2_like-dis | +8.6 | -4.3 | +21.8 |
| R3_soft | +1.0 | -12.2 | +14.6 |
| R4_rocchio | +12.1 | -1.5 | +25.9 |
| R5_naive | -1.1 | -14.6 | +11.9 |
| R8_rev_like | +8.5 | -1.8 | +20.6 |
| R9_rev | +10.9 | +2.2 | +20.0 |
| R7_lgbm | +18.5 | +7.1 | +29.5 |

control: the same rerankers with another user's persona (what any persona moves):

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm |
|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 208.0 | 211.6 | 207.3 | 213.6 | 209.9 | 201.2 | 208.4 | 220.1 |
| in_list_rank_change | 0.0 | 12.0 | 15.5 | 11.2 | 17.6 | 13.9 | 5.1 | 12.3 | 24.0 |
| in_list_moved_up | 0 | 228 | 217 | 227 | 211 | 224 | 341 | 176 | 171 |
| in_list_stayed | 514 | 7 | 6 | 6 | 3 | 5 | 13 | 3 | 3 |
| in_list_moved_down | 0 | 279 | 291 | 281 | 300 | 285 | 160 | 335 | 340 |
| in_list_top10 | 27 | 29 | 28 | 30 | 25 | 25 | 27 | 26 | 19 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 787 | 869 | 787 | 956 | 877 | 733 | 756 | 515 |
| all_mean_rank | 438.8 | 426.6 | 426.9 | 426.7 | 428.0 | 426.9 | 425.1 | 427.0 | 425.6 |

| reranker | in-list rank change, own persona | another user's persona |
|---|---|---|
| R1_like | +3.5 | +12.0 |
| R2_like-dis | +8.7 | +15.5 |
| R3_soft | +4.3 | +11.2 |
| R4_rocchio | +11.8 | +17.6 |
| R5_naive | +7.0 | +13.9 |
| R8_rev_like | +5.7 | +5.1 |
| R9_rev | +7.7 | +12.3 |
| R7_lgbm | +21.1 | +24.0 |
