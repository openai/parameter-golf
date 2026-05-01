# Reflections on LoopRes-XSA / FullAttnRes LoopQ

When I was designing this architecture, my main question was:

> How can I benefit the most from weight tying in a recurrent kind of way, but without having the model keep applying the same exact representation over and over?

The intuition was that recurrence should let a small model reuse parameters more efficiently, but naive recurrence has a risk: if I process through a layer once, then put it through again, maybe it just does the same kind of processing again. Maybe that is useful, but I do not know that. So I wanted a way for repeated applications of the same weights to be routed or weighted differently.

Maybe some representations can go one way, some can go another, or the model can learn how much to use each repetition.

That idea lent itself very naturally to attention residuals, especially full attention residuals.

## Core Idea

The architecture uses a recurrent middle section with tied/shared computation, but it does not simply feed the current hidden state through the same blocks repeatedly in isolation. Instead, it keeps residual history from different depths and loop passes, then learns how to mix those sources.

The “loopq” idea is essentially a learned query over loop/depth history:

> If I have processed this representation once, and I am about to process it again, how much should I use the current state versus earlier states?

This lets the recurrent core behave less like “run the same block again” and more like “run the same block again, but with learned access to previous intermediate representations.”

## Full Attention Residuals

I originally wanted to ablate full attention residuals against block attention residuals, but I was not able to do that in my limited time. So I should be careful here: I do not know whether full attention residuals were strictly better than block attention residuals in this setting.

My reason for trying full attention residuals was that they seemed like a natural fit for the recurrent structure. They let the model choose across a richer set of previous states. Instead of every loop pass being forced to continue from only the latest hidden representation, the model can learn a weighted mixture of previous representations.

I am still not sure whether the full version was actually worth the extra complexity, or whether block attention residuals could have achieved a similar benefit with better efficiency. That is one of the ablations I would most want to run next.

## Exclusive Self-Attention

Because the model is recurrent, I was worried that it might copy or reinforce a token’s own representation more than intended. That seemed especially plausible when the same core is applied multiple times.

So I applied exclusive self-attention because it removes the self-aligned component from the attention output, which helps free the token from relying too heavily on its own value direction. My intuition was that this would make the recurrent passes spend less capacity on self-copying and more capacity on useful cross-token processing.

That felt in line with my intentions when using recurrence: if I am going to reuse the same weights, I want each pass to do something meaningfully different.

## Unintentionally PARCAE

I also thought there needed to be some processing before and after the recurrent section, and my few experiments suggested that this helped.

So the architecture has:

- two layers in the front (called a prelude),
- two layers in the middle that get looped through 3 times (core),
- and an additional two layers at the end (coda).

`2 prelude + (2 core * 3 loops) + 2 coda = 10 effective blocks`

This is similar to the PARCAE paper. Funny enough, that paper came out after I'd played with the idea a little bit, and I renamed the components to match their terminology because it made the architecture easier to describe and easier for LLMs to reason about (especially when I was able to let them reference the paper directly).

## What I Learned

I threw these ideas together into an actual training script over the last three days, but it was a very good experience to work with larger amounts of compute and not just small-scale replications on my laptop.

This was also my first real time pre-training models from scratch. I am fully self-taught, and most of what I had done before was fine-tuning models or following fairly clear recipes. I have spent a lot of time thinking about architectures and how models should work, but this was the first time I really went off on my own and said: here is an architecture idea, now let me actually implement it and see what happens. I am still learning, so that part felt important to me.

I had never really scaled up an experiment before, so this was a wonderful learning experience. It was also very helpful to look at other people’s submissions. I initially treated the last week as a challenge to think through how I would approach this without leaning too much on existing submissions. But later it was very interesting to see how people were using different paradigms, especially test-time compute, test-time training, and more aggressive quantization.

## What I Would Push Further

I think there is a lot more to push in this architecture.

In particular:

- going wider and deeper could make the recurrent structure more useful,
- more aggressive quantization could leave room for more parameters,
- increasing model width might help with capacity,
- and the full attention residual / loop-query mechanism probably deserves more careful ablations.

I also think I could have been more aggressive with quantization and used the saved artifact budget to increase capacity. But I wanted to get a submission in, and that was important to me.

Overall, this was a useful architecture exploration: recurrent weight tying, learned routing across repeated representations, full attention residuals, exclusive self-attention, and a prelude/core/coda layout all seemed to point in the same direction.

And that is why I want to submit, even though I do not have the top record run, because there is no better way to learn than learning by doing, I suppose.
