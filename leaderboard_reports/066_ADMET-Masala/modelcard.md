# Modelling approach

Our entry to the OpenADMET CYP inhibition challenge, in outline.

## Where we started

We began from the obvious place and spent a long time there. If inhibition is a binding event,
then putting each compound into the enzyme and measuring what it does when it arrives ought to
tell you something a fingerprint cannot. We built the pipeline, docked the full library against
an ensemble of structures, and looked carefully at what came back.

It told us very little. The geometric descriptors added nothing we could measure, and the
handful of cheap structural features computed alongside them as controls carried more signal
than the poses did. We revisited that conclusion twice more from different angles later in the
campaign, including one framing designed specifically to give the pose features their best
chance, and it survived every retest. Knowing which doors are closed turned out to be worth
something, even if it was not what we were hoping to learn.

## The shape that worked

What eventually carried the entry was a different idea: do not try to model the enzyme, model
the molecule as richly as possible and let a flexible learner find the rest.

Pretrained molecular foundation models do most of the work. We leaned on encoders that read a
compound's three-dimensional shape and others that read its two-dimensional graph, adapted them
towards chemistry adjacent to our problem, and read out the internal description each one forms
of a molecule. On top of those descriptions sit modern in-context tabular learners, which are
given the labelled compounds and the unlabelled ones together and answer in a single pass
rather than being trained in the usual sense.

That split — a shared representation underneath, a separate small learner per endpoint above —
is the backbone of everything we submitted. The representations are shared across all four
enzymes; the learners are not.

## What we learned about adaptation

The most useful discovery was about what to adapt a pretrained encoder *towards*.

The instinctive answer is the thing you are scored on. That turned out to be the weaker choice
almost every time. Adapting towards a neighbouring, more abundant measurement of the same
compounds consistently produced better representations than adapting towards the scored
endpoint itself — more data, weaker labels, better encoder. Several of our strongest components
never saw the target quantity during their adaptation at all.

We also tried widening the net to large external datasets from other sources. That did not
work, and the contrast is instructive: a different *measurement* of the same compounds helped,
while different *compounds* measured elsewhere did not. The encoder spent its budget learning
the new distribution rather than sharpening its view of ours.

## Combination

Individual models plateaued well before the ensemble did, so a lot of attention went into how
to combine them. We compared a range of strategies — simple averages, selection rules, fitted
weights, and learned combiners that take the members' predictions as inputs.

The clearest finding was that the sophisticated options are not better. Several sensible
averaging rules land in a statistical dead heat, and the two approaches that try to *learn* the
combination are measurably worse, which we put down to having far fewer labelled compounds than
the combiner has freedom to exploit.

The second finding was more useful. For a long stretch, adding better models changed nothing.
What moved the number was adding models that were *wrong differently* — including several that
are poor in isolation and would never be submitted on their own. One of the weakest things we
built earns a meaningful share of the final weight purely because it disagrees with everything
else in the right places.

## Calibration

Two late adjustments mattered more than most of the modelling.

On the regression side, a single constant offset per enzyme, fitted on training data. It
changes no compound's rank and cannot improve the model, but the scoring rule treats weak and
potent compounds very differently and the offset exploits that honestly.

On the classification side, moving the decision threshold to where it belongs. Our probabilities
did not have their optimum at the default cut, and correcting that was worth more than every
modelling improvement we made on that track combined. An uncomfortable lesson, cheaply learned.

## How we kept ourselves honest

Everything above is measured on the organisers' fixed splits, out of fold. Where a procedure
involved a choice — which members, what weights, which offset, where to threshold — that choice
was made without the data it was scored on, so the quoted figures already carry the cost of
having searched. Where we could not arrange that cleanly, we have said so rather than quoting
the flattering number.

We also treat the cross-validation figures as a ranking tool rather than a forecast. The test
set was constructed differently from the training set, and we have assumed throughout that the
blind score will not match the fold score.

## Honest summary

The entry is an ensemble of pretrained molecular representations with in-context learners on
top, combined by fitted weights and calibrated against the scoring rule. Most of what we tried
did not work. The parts that did were rarely the parts we expected, and the single largest
contribution arrived very late from a colleague's parallel line of work.
