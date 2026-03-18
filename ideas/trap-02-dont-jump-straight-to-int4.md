# Trap 2. Do Not Jump Straight to Int4 or Compression Stunts

## Why It Sounds Clever

Lower precision sounds like the obvious reaction to a 16 MB artifact cap.

## Why I Think It Is a Trap Here

The baseline already fits under the cap. The problem is not “this model cannot fit.” The problem is “the submitted model loses too much quality after the current export path.”

Moving directly to int4, NF4, or a more aggressive compression stunt adds major implementation complexity and likely damages score unless the entire training/export stack is redesigned around it. That is a lot of complexity to solve the wrong first-order problem.

The higher-confidence path is:

- keep most of the current int8 machinery
- spend the remaining slack on smarter outlier handling
- or train the model to survive the exact quantization scheme

That is much more likely to improve final `val_bpb` quickly.

## Recommendation

Probably not worth trying early
