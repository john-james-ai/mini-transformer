Short answer: for sequence-to-sequence (MT) with a Transformer, **yes—people almost always include BOS/EOS on the *target***. Source BOS/EOS is optional.

### Common practice (MT, encoder–decoder)

* **Target (decoder input / labels):**

  * **Train:** input = **BOS + y₁…yₙ**, labels = **y₁…yₙ + EOS** (teacher forcing; shift by 1).
  * **Infer:** start from **BOS**, generate until **EOS** or max length.
* **Source (encoder input):**

  * Usually **no BOS/EOS**; just tokenize + pad + mask. (EOS on source is harmless but not needed.)
* **Padding:** pad IDs on both sides; mask pads everywhere; ignore pad in loss.

### Causal LMs (for comparison)

* Often **no BOS during training** (concatenate docs with a special separator); do use an **EOS / end-of-text** token to delimit sequences. At inference, some add BOS; many just start from a prompt.

### SentencePiece specifics

* SPM reserves `<s>` and `</s>`; when encoding, you can set `add_bos=True`, `add_eos=True` **for the target only**. Keep a distinct `pad_id`.
* Remember to budget tokens: if `T=256` and you use BOS+EOS on target, your **max target content length** is `254`.

### Minimal, practical recipe for your mini MT

* Reserve IDs (example): `pad=0`, `bos=1`, `eos=2` (or use your tokenizer’s actual IDs).
* **Source:** tokenize → cap to `T_src` → pad.
* **Target:** tokenize → if length > `T_tgt-2` (for BOS/EOS) **drop** the example → form `(input, label)` with BOS/EOS as above → pad.
* Loss: cross-entropy on labels with `ignore_index=pad_id`.

That’s the standard, simple setup you’ll see in most MT codebases and it works great for your toy transformer.
