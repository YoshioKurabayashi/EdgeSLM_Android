# P12e numeric evidence

All errors are relative L2 against prepped FP32, averaged per token.

## chat_original (107 tokens)

| Variant | Later top1 | Response prediction top1 | Later add53 | Later logits | Later KL | New K | New V |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 81/106 (76.4%) | 26/29 | 29.67% | 36.96% | 0.6637 | 17.61% | 37.71% |
| shared | 0/106 (0.0%) | 0/29 | 2033.16% | 277.06% | 10.5208 | 84.29% | 130.44% |
| split | 3/106 (2.8%) | 0/29 | 107.82% | 228.20% | 8.7356 | 72.47% | 133.37% |
| focused_shared | 72/106 (67.9%) | 20/29 | 38.12% | 41.63% | 0.8765 | 22.08% | 48.52% |
| focused_split | 86/106 (81.1%) | 26/29 | 23.44% | 34.52% | 0.6402 | 14.91% | 31.22% |

## chat_english (109 tokens)

| Variant | Later top1 | Response prediction top1 | Later add53 | Later logits | Later KL | New K | New V |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 91/108 (84.3%) | 26/29 | 27.40% | 40.98% | 0.4573 | 16.56% | 35.53% |
| shared | 8/108 (7.4%) | 0/29 | 7465.30% | 188.95% | 10.7510 | 106.13% | 105.30% |
| split | 2/108 (1.9%) | 0/29 | 106.65% | 282.51% | 8.6814 | 73.43% | 133.83% |
| focused_shared | 87/108 (80.6%) | 25/29 | 35.17% | 45.37% | 0.5330 | 20.42% | 44.57% |
| focused_split | 92/108 (85.2%) | 26/29 | 20.95% | 36.10% | 0.3713 | 14.18% | 29.09% |

## Free generation, original Japanese dialogue

Temperature -> music -> recall temperature.

### fp32

Turn 1; tokens=11; stop=151645
```text
エアコンを22度に設定できます。
```

Turn 2; tokens=6; stop=151645
```text
音楽をかけてます。
```

Turn 3; tokens=9; stop=151645
```text
さっきは22度にしてました。
```

### baseline

Turn 1; tokens=12; stop=151645
```text
エアコンを22度に調節しました。
```

Turn 2; tokens=6; stop=151645
```text
音楽をかけてます。
```

Turn 3; tokens=9; stop=151645
```text
さっきも22度にしました。
```

### shared

Turn 1; tokens=32; stop=None
```text
"

InInIn



InInIn




InInIn
IncessiveIntIn





Inicient

icient


```

Turn 2; tokens=32; stop=None
```text
InInicientInIn




InInicient




InicientInInInInInInInIn





InicientInInIn
```

Turn 3; tokens=32; stop=None
```text
InInInInInInInInInInInInInInInInInInInInInInIncessiveInInInInInInInIn
```

### split

Turn 1; tokens=32; stop=None
```text






 "

"""

"

"





."

"

"

"

"

."

"

"

"

."."


.Es


```

Turn 2; tokens=32; stop=None
```text
""""""""

"""

"

"

.E".Integer"

"

"

"

.E.Integer"."

"

"

"

"

"

"

"

"


```

Turn 3; tokens=32; stop=None
```text


...

...

""...

In""

"""

"...

"

"

"

""

"

In"""Internal"
```

### focused_shared

Turn 1; tokens=17; stop=151645
```text
エアコンを22度にすると良いです。何か他の設定ですか？
```

Turn 2; tokens=17; stop=151645
```text
えっれー！音楽をかけていてよ！何か他の設定ですか？
```

Turn 3; tokens=15; stop=151645
```text
さっき何度にかけても良いです。何か他の設定ですか？
```

### focused_split

Turn 1; tokens=21; stop=151645
```text
エアコンを22度にすると、室内の気臭さが少なくなりますよ。
```

Turn 2; tokens=21; stop=151645
```text
エアコンを22度にすると、室内の気臭さが少なくなりますよ。
```

Turn 3; tokens=11; stop=151645
```text
さっきは22度にしてしまったよ。
```

### hybrid

Turn 1; tokens=32; stop=None
```text
エアコンを22度にすると、室内の気臭さが減って良いですね。何か他の設定に合わせて調整してみてください。
```

Turn 2; tokens=32; stop=None
```text
エアコンを22度にすると、室内の気臭さが減って良いですね。何か他の設定に合わせて調整してみてください。
```

Turn 3; tokens=21; stop=151645
```text
さっきも22度にしていましたね。何か他の設定に合わせて調整してみてください。
```
