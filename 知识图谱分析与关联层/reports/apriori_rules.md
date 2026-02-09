# Apriori 关联规则（Top）

- Transactions: 946
- Frequent itemsets: 2183 (min_support=0.02, max_len=3)
- Rules saved: 50 (min_confidence=0.3)

| # | Antecedent | Consequent | Support | Confidence | Lift |
|---:|---|---|---:|---:|---:|
| 1 | painpoint=handle_comfort_positive | painpoint=sharpness_positive | 0.0211 | 1.0000 | 14.7812 |
| 2 | has_aplus=1; painpoint=handle_comfort_positive | painpoint=sharpness_positive | 0.0211 | 1.0000 | 14.7812 |
| 3 | painpoint=handle_comfort_positive; rating>=4.5 | painpoint=sharpness_positive | 0.0201 | 1.0000 | 14.7812 |
| 4 | painpoint=sharpness_positive | painpoint=handle_comfort_positive | 0.0211 | 0.3125 | 14.7812 |
| 5 | has_aplus=1; painpoint=sharpness_positive | painpoint=handle_comfort_positive | 0.0211 | 0.3125 | 14.7812 |
| 6 | knife_type=chef; painpoint=sharpness_positive | painpoint=overall_quality_positive | 0.0201 | 0.3276 | 14.7570 |
| 7 | painpoint=sharpness_positive; rating>=4.5 | painpoint=handle_comfort_positive | 0.0201 | 0.3065 | 14.4952 |
| 8 | painpoint=overall_quality_positive | painpoint=sharpness_positive | 0.0201 | 0.9048 | 13.3735 |
| 9 | has_aplus=1; painpoint=overall_quality_positive | painpoint=sharpness_positive | 0.0201 | 0.9048 | 13.3735 |
| 10 | knife_type=chef; painpoint=overall_quality_positive | painpoint=sharpness_positive | 0.0201 | 0.9048 | 13.3735 |
| 11 | knife_type=santoku; material=damascus | material=vg10 | 0.0222 | 0.4773 | 9.0300 |
| 12 | knife_type=chef; material=damascus | material=vg10 | 0.0507 | 0.3810 | 7.2076 |
| 13 | is_fba=1; material=damascus | material=vg10 | 0.0433 | 0.3761 | 7.1167 |
| 14 | material=damascus; rating>=4.5 | material=vg10 | 0.0423 | 0.3540 | 6.6973 |
| 15 | knife_type=santoku; material=vg10 | material=damascus | 0.0222 | 1.0000 | 6.6620 |
| 16 | material=vg10 | material=damascus | 0.0507 | 0.9600 | 6.3955 |
| 17 | has_aplus=1; material=vg10 | material=damascus | 0.0507 | 0.9600 | 6.3955 |
| 18 | knife_type=chef; material=vg10 | material=damascus | 0.0507 | 0.9600 | 6.3955 |
| 19 | material=damascus | material=vg10 | 0.0507 | 0.3380 | 6.3955 |
| 20 | has_aplus=1; material=damascus | material=vg10 | 0.0507 | 0.3380 | 6.3955 |
| 21 | knife_type=set; material=vg10 | material=damascus | 0.0243 | 0.9583 | 6.3844 |
| 22 | is_fba=1; material=vg10 | material=damascus | 0.0433 | 0.9535 | 6.3521 |
| 23 | material=vg10; rating>=4.5 | material=damascus | 0.0423 | 0.9524 | 6.3447 |
| 24 | knife_type=set; material=damascus | material=vg10 | 0.0243 | 0.3194 | 6.0439 |
| 25 | knife_type=boning; material=carbon_steel | knife_type=cleaver | 0.0222 | 0.7241 | 5.9055 |
| 26 | knife_type=cleaver; material=high_carbon | knife_type=nakiri | 0.0233 | 0.3143 | 4.5741 |
| 27 | knife_type=nakiri; material=high_carbon | knife_type=cleaver | 0.0233 | 0.5500 | 4.4853 |
| 28 | knife_type=nakiri; rating>=4.5 | knife_type=cleaver | 0.0254 | 0.5217 | 4.2549 |
| 29 | knife_type=nakiri | knife_type=cleaver | 0.0338 | 0.4923 | 4.0149 |
| 30 | has_aplus=1; knife_type=nakiri | knife_type=cleaver | 0.0338 | 0.4923 | 4.0149 |
| 31 | knife_type=chef; knife_type=nakiri | knife_type=cleaver | 0.0317 | 0.4918 | 4.0107 |
| 32 | is_fba=1; knife_type=nakiri | knife_type=cleaver | 0.0285 | 0.4909 | 4.0034 |
| 33 | knife_type=cleaver; material=high_carbon | material=carbon_steel | 0.0455 | 0.6143 | 3.6094 |
| 34 | knife_type=cleaver; knife_type=set | knife_type=boning | 0.0285 | 0.4426 | 3.5187 |
| 35 | knife_type=cleaver; material=carbon_steel | knife_type=boning | 0.0222 | 0.4286 | 3.4070 |
| 36 | knife_type=boning; knife_type=cleaver | material=carbon_steel | 0.0222 | 0.5526 | 3.2471 |
| 37 | knife_type=boning; rating>=4.5 | knife_type=cleaver | 0.0349 | 0.3837 | 3.1293 |
| 38 | knife_type=cleaver; rating>=4.5 | material=carbon_steel | 0.0497 | 0.5222 | 3.0685 |
| 39 | knife_type=boning; material=high_carbon | knife_type=cleaver | 0.0254 | 0.3692 | 3.0111 |
| 40 | knife_type=cleaver; rating>=4.5 | knife_type=boning | 0.0349 | 0.3667 | 2.9148 |
| 41 | material=carbon_steel; rating>=4.5 | knife_type=cleaver | 0.0497 | 0.3561 | 2.9037 |
| 42 | knife_type=boning; knife_type=chef | knife_type=cleaver | 0.0381 | 0.3462 | 2.8229 |
| 43 | knife_type=chef; knife_type=cleaver | knife_type=boning | 0.0381 | 0.3495 | 2.7785 |
| 44 | material=carbon_steel; material=high_carbon | knife_type=cleaver | 0.0455 | 0.3386 | 2.7612 |
| 45 | knife_type=cleaver; material=high_carbon | knife_type=boning | 0.0254 | 0.3429 | 2.7256 |
| 46 | knife_type=cleaver; knife_type=set | material=carbon_steel | 0.0296 | 0.4590 | 2.6971 |
| 47 | is_fba=1; knife_type=boning | knife_type=cleaver | 0.0317 | 0.3226 | 2.6307 |
| 48 | knife_type=chef; knife_type=cleaver | material=carbon_steel | 0.0486 | 0.4466 | 2.6241 |
| 49 | knife_type=cleaver | knife_type=boning | 0.0402 | 0.3276 | 2.6042 |
| 50 | has_aplus=1; knife_type=cleaver | knife_type=boning | 0.0402 | 0.3276 | 2.6042 |
