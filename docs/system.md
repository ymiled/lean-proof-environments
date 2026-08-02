---
title: "proof-depth-decay"
subtitle: "How the system works, and why each piece is shaped the way it is"
date: "August 2026"
geometry: margin=1in
fontsize: 11pt
mainfont: "Palatino"
monofont: "Menlo"
colorlinks: true
linkcolor: MidnightBlue
urlcolor: MidnightBlue
toc: true
toc-depth: 2
---

\newpage

# What this is, in one page

> **Note (August 2026).** This document describes the system at the time three
> families shipped. The `arithmetic` family has since been removed and the
> benchmark narrowed to information-flow security; see `docs/design-log.md`.
> Sections referring to the arithmetic ladder are retained as design history.

A **Lean 4 environment for measuring how a language model's proof success falls
off as proofs get deeper**, plus the experiments run against it.

It exists to test one published claim. Theorem's `lf-lean` post reports that
without compositional structure, success drops by "roughly a 3x reduction in
successful problem count for every additional dependency solved." That constant
is the justification for treating already-verified dependencies as trusted
interfaces rather than re-proving them, which is the central architectural
decision of their pipeline. It was measured once, on one corpus, with one
pipeline, and has not been independently checked.

The system has four parts:

1. **A ladder.** A dependency graph of lemmas over a small Lean theory, where
   lemma $L$ at depth $d$ sits on a chain of $d-1$ lemmas beneath it.
2. **Two conditions.** Either the ancestors are handed to the model already
   proved (*compositional*), or they are withheld and must be re-proved
   (*monolithic*).
3. **A grader.** Runs the Lean kernel and audits which axioms the result depends
   on. Binary, objective, no learned reward model.
4. **A harness.** Renders tasks, collects answers from a policy, grades, and
   reports.

About 2,100 lines of Python, plus Lean source that is generated rather than
checked in.

**It is not a training run.** There is no gradient, no policy update. This is the
*environment half* of an RL setup, with a frozen model as the policy. `env.py`
exposes `reset`/`step` so a learner could be attached, but attaching one is out
of scope.

\newpage

# Part I: The core idea

## Dependency depth

Every lemma gets a depth: 1 if it is provable from the definitions alone,
otherwise $1 + \max(\text{depth of its dependencies})$. Depth is *computed from
the graph*, never declared, so a lemma cannot misreport its own difficulty.

Take the `arithmetic` family. Writing `op1` as $+$ and `op2` as $\times$ (both
are randomly renamed in the actual tasks):

| depth | lemma | statement | needs |
|---|---|---|---|
| 1 | `add_zero_l` | $0 + y = y$ | nothing |
| 1 | `add_succ_l` | $(x{+}1) + y = (x{+}y){+}1$ | nothing |
| 1 | `add_assoc` | $(x{+}y)+z = x+(y{+}z)$ | nothing |
| 2 | `add_comm` | $x + y = y + x$ | `add_zero_l`, `add_succ_l` |
| 3 | `add_left_comm` | $x+(y{+}z) = y+(x{+}z)$ | `add_assoc`, `add_comm` |
| 4 | `mul_succ_l` | $(x{+}1)\times y = x{\times}y + y$ | `add_left_comm`, … |
| 5 | `mul_comm` | $x \times y = y \times x$ | `mul_zero_l`, `mul_succ_l` |

Read bottom up: to prove $x \times y = y \times x$ you first need
$(x{+}1)\times y = x{\times}y + y$, which needs $x+(y{+}z) = y+(x{+}z)$, which
needs $x+y = y+x$, which needs $0+y = y$. Four lemmas stacked. Depth 5.

The operations recurse on their **second** argument, which is why the
left-handed statements need induction instead of being true by computation. That
asymmetry is the entire source of the chain.

## The noninterference theory, from first principles

The second family formalises a classical result in language-based security. This
section explains the mathematics; the next lists the rungs that build it.

### The security property

Split every variable into **public** and **secret**. An attacker can read the
public variables and nothing else. The question: can running a program leak
information from secret variables into public ones?

The formal answer is stated as an indistinguishability property. Two starting
states that the attacker cannot tell apart must produce two ending states the
attacker cannot tell apart. If that holds, watching the public variables tells
the attacker nothing about the secrets.

$$
s \approx_L t \;\;\wedge\;\; \vdash c \quad\Longrightarrow\quad
\mathsf{run}(c, s) \;\approx_L\; \mathsf{run}(c, t)
$$

where $s \approx_L t$ means "$s$ and $t$ agree on all public variables" and
$\vdash c$ means "$c$ is well-typed by the security type system."

This is **noninterference**: the secret inputs do not interfere with the public
outputs.

### Why a type system, and what makes it subtle

The naive rule is "never assign a secret expression to a public variable." That
catches the **explicit flow**:

```
public_x := secret_y        -- obviously leaks
```

But it misses the **implicit flow**, which leaks through control flow rather
than through data:

```
if secret_y = 0 then public_x := 1 else public_x := 2
```

No secret value is ever assigned to `public_x`. Yet afterwards, reading
`public_x` tells you whether `secret_y` was zero. The leak travels through
*which branch ran*.

Handling implicit flows is the entire reason the type system carries a **program
counter level** `pc`, and the reason the proof needs a confinement lemma.

### The language

Expressions and commands, both as Lean inductive types:

```lean
inductive Exp where
  | lit  (n : Nat)          -- literal
  | var  (x : Nat)          -- variable, identified by a number
  | eadd (a b : Exp)        -- addition

inductive Com where
  | cskip                          -- do nothing
  | cassign (x : Nat) (e : Exp)    -- x := e
  | cseq    (c d : Com)            -- c ; d
  | cite    (e : Exp) (c d : Com)  -- if e then c else d
```

Deliberately **loop-free**. Adding `while` would force termination reasoning,
which contributes nothing to the noninterference argument and a great deal of
work.

### States, contexts, and semantics

```lean
abbrev St  := Nat → Nat     -- a state maps each variable to its value
abbrev Ctx := Nat → Bool    -- a context labels each variable: true = secret
```

A state is a *function*, which is why updating it is function override:

```lean
def upd (s : St) (x v : Nat) : St := fun y => if y = x then v else s y
```

Evaluation is a **total function**, not an inductive relation. This is the
single most important implementation choice in the family: it means every proof
proceeds by structural induction plus `simp`, rather than by induction over
derivation trees, and so stays inside a small tactic vocabulary with no Mathlib.

```lean
def evalE (s : St) : Exp → Nat
  | .lit n    => n
  | .var x    => s x
  | .eadd a b => evalE s a + evalE s b

def evalC (s : St) : Com → St
  | .cskip       => s
  | .cassign x e => upd s x (evalE s e)
  | .cseq c d    => evalC (evalC s c) d
  | .cite e c d  => if evalE s e = 0 then evalC s d else evalC s c
```

### The security lattice

Two levels only: public and secret, ordered public $\sqsubseteq$ secret. In Lean
they are `false` and `true`, so the lattice join is just boolean `||`.

An expression is secret if *any* variable in it is secret:

```lean
def lvl (G : Ctx) : Exp → Bool
  | .lit _    => false                   -- constants are public
  | .var x    => G x                     -- a variable's own label
  | .eadd a b => lvl G a || lvl G b      -- join
```

### The typing rules

`wt G pc c` is `true` when command `c` is safe to run in a context whose program
counter level is `pc`. Note it is **`Bool`-valued, not an inductive judgment** —
the second key implementation choice, since it makes typing *computable* and
lets `simp` take a typing hypothesis apart.

```lean
def wt (G : Ctx) (pc : Bool) : Com → Bool
  | .cskip       => true
  | .cassign x e => !(lvl G e || pc) || G x
  | .cseq c d    => wt G pc c && wt G pc d
  | .cite e c d  => wt G (pc || lvl G e) c && wt G (pc || lvl G e) d
```

Two rules carry all the content.

**Assignment.** `!(lvl G e || pc) || G x` reads as an implication:

$$
(\mathrm{lvl}(e) \sqcup pc) \sqsubseteq G(x)
$$

"If the expression is secret, *or* we are executing inside a secret branch, then
the assigned variable must be secret." The `lvl G e` half blocks explicit flows.
The `pc` half blocks implicit ones.

**Conditional.** Both branches are typed at `pc || lvl G e`, so branching on a
secret expression **raises the program counter to secret** for the whole body.
Combined with the assignment rule, that forbids any public assignment inside a
secret branch, which is exactly what makes the earlier `if secret_y = 0` example
ill-typed.

### The observation relation

```lean
def lowEq (G : Ctx) (s t : St) : Prop := ∀ x, G x = false → s x = t x
```

"$s$ and $t$ agree on every public variable." This is what the attacker can see,
and the whole theorem is stated in terms of it.

### The shape of the proof

The soundness proof is induction over commands. Three cases are routine, and one
is not.

*Skip* is immediate. *Assignment* needs the observation that if the target is
public then typing forces the expression to be public, so both runs compute the
same value. *Sequencing* chains the two induction hypotheses.

The **conditional with a secret guard** is the hard case, and it is where
`confinement` earns its place. The guard is secret, so the two runs may evaluate
it differently and therefore **take different branches**. There is no induction
hypothesis relating two *different* commands, so the argument cannot proceed by
comparing the branches.

Instead you argue that neither run moved the public state at all. Both branches
are typed at $pc = \texttt{secret}$, so confinement gives

$$
s \approx_L \mathsf{run}(c, s)
\qquad\text{and}\qquad
t \approx_L \mathsf{run}(c, t)
$$

and then the result follows by chaining with the assumption $s \approx_L t$,
using symmetry and transitivity:

$$
\mathsf{run}(c, s) \;\approx_L\; s \;\approx_L\; t \;\approx_L\;
\mathsf{run}(c, t)
$$

That chain is the rung `ite_high_ni`, and it is why `lowEq_symm` and
`lowEq_trans` exist as rungs at all: they are not decoration, they are the glue
for this case.

## The noninterference ladder

The second family, and the one carrying the headline result. A loop-free
imperative language with a two-point security lattice, public and secret. The
top rung is the soundness theorem: a well-typed command run from two states that
agree on public variables produces two states that still agree, so nothing flows
from secret to public.

| depth | rung | LoC | what it says | needs |
|---|---|---|---|---|
| 1 | `lowEq_refl` | 1 | public agreement is reflexive | — |
| 1 | `lowEq_symm` | 1 | public agreement is symmetric | — |
| 1 | `lowEq_trans` | 1 | public agreement is transitive | — |
| 1 | `conf_ite` | 3 | conditional step of confinement | — |
| 1 | `upd_pair` | 4 | agreement survives a common update | — |
| 1 | `assign_conf` | 5 | a secret-context assignment cannot touch a public variable | — |
| 1 | `evalE_agree` | 6 | public expressions evaluate alike in agreeing states | — |
| 1 | `wt_anti` | 18 | typing is antitone in the pc level | — |
| 2 | `assign_ni` | 7 | assignment case of the top theorem | `evalE_agree` |
| 2 | `confinement` | 11 | secret-context code leaves the public projection fixed | `lowEq_refl`, `assign_conf`, `lowEq_trans`, `conf_ite` |
| 3 | `ite_high_ni` | 3 | secret-guard conditional case | `conf_ite`, `confinement`, `lowEq_trans`, `lowEq_symm` |
| 4 | `noninterference` | 22 | well-typed programs leak nothing | `assign_ni`, `evalE_agree`, `ite_high_ni` |

### What each rung says

**`lowEq_refl`, `lowEq_symm`, `lowEq_trans`** — public agreement is an
equivalence relation. One line each, since `lowEq` unfolds to a $\forall$ and
the work is done by `Eq`'s own reflexivity, symmetry and transitivity. They look
trivial and are not optional: symmetry and transitivity are precisely the glue
in the secret-guard case.

**`upd_pair`** — if $s \approx_L t$ then $s[x \mapsto v] \approx_L t[x \mapsto v]$.
Writing the *same* value into both states preserves agreement. Proof splits on
whether the observed variable is the updated one: if yes both sides are $v$, if
no both sides are unchanged.

**`assign_conf`** — a secret-context assignment cannot disturb the public
projection: $s \approx_L s[x \mapsto v]$ when typing forced $G(x) = \texttt{secret}$.
The proof extracts $G(x) = \texttt{true}$ from the typing hypothesis, so any
public $y$ must differ from $x$, so `upd` leaves it alone.

**`evalE_agree`** — a *public* expression evaluates identically in agreeing
states: $\mathrm{lvl}(e) = \texttt{public} \wedge s \approx_L t \Rightarrow
\mathsf{run}(e, s) = \mathsf{run}(e, t)$. Induction over the
expression; the variable case is exactly the definition of $\approx_L$, and the
addition case needs both subexpressions public, which is what the `||` in `lvl`
gives.

**`wt_anti`** — typing is antitone in the program counter: anything well-typed
at $pc = \texttt{secret}$ is well-typed at $pc = \texttt{public}$. Intuitively,
the secret context is the *more* restrictive one. This is the longest depth-1
proof at 18 lines because it needs induction over commands with a case split on
the guard level in the conditional case.

**`conf_ite`** — if both branches individually leave the public projection
fixed, so does the conditional. Just a case split on which branch runs. Extracted
so that `confinement` does not have to inline it.

**`confinement`** — the key auxiliary theorem: **code typed at $pc =
\texttt{secret}$ never changes the public projection of the state.** Induction
over commands, using `assign_conf` for assignment, `lowEq_trans` to chain the
two halves of a sequence, and `conf_ite` for the conditional. This is what makes
implicit flows safe.

**`assign_ni`** — the assignment case of the top theorem, on its own. Splits on
whether the observed variable is the assigned one. If it is, typing forces the
expression public, so `evalE_agree` says both runs wrote the same value. If it
is not, the update is invisible and the assumption carries over.

**`ite_high_ni`** — the secret-guard conditional case, and where the real content
sits. The two runs may take **different branches**, so no induction hypothesis
applies. Instead confinement is applied to each side separately and the results
glued with symmetry and transitivity, per the chain in the previous section.
Three lines, because all the work has been pushed into its dependencies.

**`noninterference`** — the soundness theorem. Induction over commands: skip is
immediate, assignment defers to `assign_ni`, sequencing chains the two induction
hypotheses, and the conditional splits on the guard level. Public guard means
`evalE_agree` shows both runs take the *same* branch, so an induction hypothesis
applies. Secret guard defers to `ite_high_ni`.

Three design choices keep this tractable without Mathlib. Semantics is a
**total function** rather than an inductive relation, so rungs are discharged by
structural induction plus `simp` instead of induction over derivation trees.
Typing is **`Bool`-valued** rather than an inductive judgment, so a hypothesis
`wt G pc c = true` can be taken apart by `simp`. And the language is
**loop-free**, since `while` forces termination reasoning that adds nothing to
the noninterference argument.

### Four of these rungs were added deliberately, and are why the family works

The family originally had seven rungs and produced nothing: it floored at 0.00
above depth 1, and the chain/antichain contrast was structurally impossible in
it. Four additions fixed both problems.

* `conf_ite` and `assign_ni` **extract cases from the original 37-line soundness
  proof**, dropping the top rung to 22 lines. This alone took `noninterference`
  from unprovable to provable for the same policy at the same budget.
* `wt_anti` is **18 lines at depth 1**, a shallow-but-long rung.
* `ite_high_ni` is **3 lines at depth 3**, a deep-but-short rung.

Those last two exist to attack the confound. Depth and proof length were
correlated at +0.97 in this family, meaning any decay curve could be reread as a
length effect. Adding a long shallow rung and a short deep one dropped it to
**+0.56**, which is what makes depth and length separable here at all.

`upd_pair` and `wt_anti` are also **maximal** — nothing depends on them.
Before they existed, `noninterference` was the only maximal rung, making the
family a funnel (Part V explains why that forbids antichains entirely).

## The two conditions

| condition | what the model receives |
|---|---|
| `compositional` | the target only; every ancestor supplied, already proved, citable by name |
| `monolithic` | the target *and every ancestor* as goals; nothing supplied |

At depth 5 that is the difference between writing three lines and writing eight
lemmas from scratch.

Both conditions are cases of one general form: **a task is a set of lemmas to
prove, and everything else in the family is supplied.** Compositional is the set
$\{L\}$; monolithic is $\{L\} \cup \text{ancestors}(L)$. Arbitrary sets in
between are what the chain/antichain contrast uses (Part V).

\newpage

# Part II: Architecture

## Data flow

```
  seed -> Instance.sample(family, seed)        ladder.py
            |   fresh identifiers for the whole theory
            v
          Task(instance, targets)              task.py
            |- .preamble()   definitions + supplied lemmas WITH proofs
            |- .header()     "theorem lem_a7f2 (x y : Warp) : ... := by"
            `- .prompt()     what the policy sees
            v
          Policy.act(task) -> tactic blocks    policy.py / agentrun.py
            v
          Task.assemble(blocks) -> a complete .lean file
            v
          grade() -> lean subprocess -> Verdict   grader.py
            v
          report / sweep -> results/*.json -> plot
```

One direction of dependency throughout: `ladder → task → grader → {policy, env,
sweep} → report`. No cycles.

## Modules

| file | lines | role |
|---|---|---|
| `ladder.py` | 159 | `Rung`, `Family`, `Instance`; depth derivation; renaming |
| `families/vsi.py` | 84 | machine-checked VSI chain, 14 rungs, depths 1–5 |
| `families/noninterference.py` | 317 | security type system, 12 rungs, depths 1–4 |
| `task.py` | 234 | target sets, both conditions, prompts, assembly |
| `grader.py` | 145 | compile, axiom audit, banned syntax |
| `selftest.py` | 78 | certifies a family before any sweep is trusted |
| `design.py` | 148 | chain/antichain contrast enumeration |
| `policy.py` | 100 | reference / empty / API policies |
| `agentrun.py` | 268 | export/ingest split, leak-free batching |
| `env.py` | 83 | gym-style `reset`/`step` |
| `sweep.py` | 161 | experiment driver |
| `report.py` | 90 | result tables with proof lengths |
| `plot.py` | 117 | pass@k vs depth, decay fit |

The organising principle: **`ladder.py` knows no mathematics, and the families
are the only files that do.** Everything downstream consumes `DEFINITIONS`,
`RUNGS`, and the dependency graph through one interface, so adding a theory is
one new module and no edits anywhere else.

## A rung is a schema, not text

```python
Rung(
    key="add_comm",
    binders="(x y : {T})",
    statement="{op1} x y = {op1} y x",
    proof="  induction y with\n"
          "  | {zero} => simp [{op1}, {add_zero_l}]\n"
          "  | {succ} k ih => simp [{op1}, {add_succ_l}, ih]",
    deps=("add_zero_l", "add_succ_l"),
)
```

Every identifier is a placeholder. `Instance.sample(family, seed)` draws fresh
names (`Warp`, `braid`, `lem_a7f2`) and formats the schemas against them, so
renaming costs nothing and every seed produces a fresh-looking theory.

\newpage

# Part III: The grader

This is the part that has to be right. A verification reward is worth exactly as
much as its grader, and Lean offers several ways to close a goal without proving
it.

## The incident that determined the design

While building the arithmetic ladder, one rung's proof failed. Lean reported the
error, **but still declared the theorem**. Every downstream lemma citing it
compiled with exit status 0 and no diagnostics of its own. Only this revealed the
problem:

```
'L07' depends on axioms: [propext, sorryAx]
```

A grader asking "did the file compile?" would have scored that entire subtree as
proved. So the acceptance test is the axiom audit, not the exit code.

## What acceptance requires

1. **Compiles**, exit 0, no `error:` diagnostics.
2. **`#print axioms` on every target** is a subset of
   `{propext, Classical.choice, Quot.sound}`. This is the load-bearing check. It
   catches `sorry`, `admit`, any freshly declared axiom, and anything inherited
   transitively through a cited lemma.
3. **No banned syntax** in the model's own text: `sorry`, `admit`,
   `native_decide`, `axiom`, `unsafe`, `partial`. `native_decide` is excluded
   because it trusts the compiler rather than the kernel, and so leaves no axiom
   trace of the kind we scan for.

Multi-target tasks are graded **all or nothing**. Every target must clear every
check. That matches the quantity under test, which is a count of problems
solved, and it stops a policy scoring by proving only the easy members of a set.

## Statement drift is impossible, not merely detected

The policy is asked for a **tactic block only**, never a whole file. The harness
writes the `theorem ... := by` header itself and splices the model's text
underneath.

This is not a convenience. A model cannot weaken the goal, restate it, or prove
a different lemma under a matching name, because it never gets to write the
statement. That removes the largest class of reward hacking and needs no
statement-comparison code at all. An earlier design compared statements up to
definitional equality; it was deleted in favour of this.

\newpage

# Part IV: The three certifications

`selftest.py` must report zero failures before any sweep is trusted. It checks
three things per rung, and each one exists because an earlier ladder passed the
others while being broken.

## 1. Solvable compositionally

The reference proof, with ancestors supplied, must be accepted by the grader.
Obvious, but it catches families that stopped compiling after an edit.

## 2. Solvable monolithically

The monolithic reference proof re-proves every ancestor inline as a `have`
before the target. Without this check, a decay curve could equally well be
reporting that the tasks were *impossible*, and the experiment would be
measuring the benchmark rather than the model.

## 3. Depth is real

The **citing** proof must *fail* when its ancestors are removed. A dependency
that survives removal is decorative and the rung measures nothing.

This check exists because of a ladder that passed checks 1 and 2 while having no
real depth at all. The lemmas were stated at a concrete point
(`op_i one y = one`). Every such goal is a **closed term**, so `simp` does not
reason about it, it *evaluates* it:

```lean
theorem L3_alone (y : Tally) : op3 one y = one := by
  induction y with
  | zero => rfl
  | succ k ih => simp_all [one, op1, op2, op3]
-- compiles, with zero lemmas in scope
```

A depth-3 rung fell with nothing in scope. The rule extracted: **lemmas must be
universally quantified over the variables the definitions recurse on**, so
residual goals contain free variables, reduction is blocked, and induction is
genuinely required.

Verified on the fixed version, with `add_left_comm` withheld but `add_assoc`,
`add_comm` and full automation available:

```
error: unsolved goals
|- op1 x (op1 k (op2 x k)) = op1 k (op1 x (op2 x k))
```

That residual *is* the withheld lemma. No way around it.

\newpage

# Part V: The two experiments

## Experiment 1: depth sweep

Every rung, both conditions, pass rate against depth. Directly measures the
quantity Theorem report.

## Experiment 2: chain versus antichain

The depth sweep alone cannot support the conclusion you want from it, because
depth, dependency count and total proof length all move together. Measured on
the arithmetic ladder, corr(depth, monolithic length) $\approx 0.98$.

So if pass rate falls with depth, at least two incompatible stories fit:

* **Depth is causal.** Chains compound: each step is conditional on the previous
  one being right.
* **Volume is causal.** Deep tasks simply require more output, and depth is only
  a proxy.

These have different consequences. Under the second, compositional verification
helps for reasons unrelated to dependency structure and any chunking would do.
And it is not a strawman: Theorem's own post reports a difficulty cliff at
"17+ marginal LoC", and in the arithmetic ladder monolithic length crosses 17
right between depth 3 and depth 4. **Their length cliff predicts our curve with
no depth effect existing at all.**

### The contrast

Compare two task sets of the **same size** and comparable total length:

* **antichain** — $k$ mutually independent lemmas. Volume $k$, no chaining.
* **chain** — $k$ lemmas where each feeds the next. Volume $k$, maximal chaining.

Equal pass rates implicate volume. A gap implicates chaining.

### Two structural constraints, both found the hard way

**Withheld sets must be up-closed.** A supplied lemma arrives with its reference
proof, which cites its own dependencies, so those must be supplied too.
Withholding a lemma therefore forces withholding everything above it. The
consequence: **antichains can only be drawn from maximal elements** — rungs that
nothing depends on. A family with one maximal rung is a *funnel* and cannot
support the contrast at all.

**Uniformity destroys essential depth.** A purpose-built family was written to
make the two arms match exactly, with every rung the same shape. It failed: each
rung was the previous one composed once more, so the chain could be bypassed
entirely by unfolding definitions and reusing the base lemma.

```lean
theorem C4_nochain (x y : N) : f4 (ad x y) = ad (f4 x) (f4 y) := by
  simp only [f4, f3, f2]
  rw [W1, W1, W1, W1]
-- compiles with C2 and C3 absent
```

Matching by construction requires uniformity; uniformity makes the chain
decorative. The two requirements are contradictory, so matched pairs must be
*searched for* among naturally occurring rungs instead. `design.py` does that,
and finds exactly LoC-matched pairs (6 vs 6 at $k=2$, 9 vs 9 at $k=3$).

\newpage

# Part VI: Running it, and the leak

## Export and ingest

A policy without an HTTP API can sit in the middle of a sweep:

```bash
uv run python -m pdd.agentrun export --family noninterference --mode depth
uv run python -m pdd.agentrun plan       # who answers what
# ... a policy writes runs/<id>.out for each runs/<id>.txt ...
uv run python -m pdd.report --label ni-sonnet5
```

## Why batching is dangerous

A **compositional prompt supplies the target's ancestors together with their
proofs.** So if one policy instance holds two compositional tasks and one target
is an ancestor of the other, the second prompt hands over the first answer.

This is not hypothetical. In the first run, agents batched across conditions
copied ancestor proofs out of compositional prompts into their monolithic
answers, and reported doing so.

The rule: monolithic prompts contain no proofs at all and may share a batch
freely; compositional tasks are grouped **by depth**, because two rungs at equal
depth cannot be ancestors of one another (an ancestor has strictly smaller
depth), so a depth level is an antichain.

`verify_groups` checks the property directly rather than trusting the argument,
and `export` refuses to write files if it fails. Against the naive single-batch
grouping it reports ten violations, naming each one.

**And it was still violated twice by hand** after being verified, by merging
batches to save agents. Hence `agentrun plan`, which emits the assignment so it
is not left to judgment.

\newpage

# Part VII: Results so far

## Arithmetic, depth sweep

| depth | mono lemmas | Sonnet 5 | Haiku 4.5 |
|---|---|---|---|
| 1 | 1 | 1.00 | 1.00 |
| 2 | 2–3 | 1.00 | 0.75 |
| 3 | 3–5 | 1.00 | 0.33 |
| 4 | 5–6 | 1.00 | 0.00 |
| 5 | 8 | 0.00 | 0.00 |

Sonnet compositional: 15/15, flat at every depth.

Fitting log(pass) against depth on the non-zero Haiku points gives **1.73x per
depth**, $R^2 = 0.93$, against the reported ~3x.

**But the same data fitted against lemma count gives 1.44x per lemma at
$R^2 = 0.930$, identical to three decimals.** The two models are
indistinguishable. The number is a measurement, not a mechanism, and not a
replication.

## Noninterference, depth sweep

| depth | compositional | monolithic | mono lemmas |
|---|---|---|---|
| 1 | 1.00 (8) | 0.75 (8) | 1 |
| 2 | 0.50 (2) | 0.00 (2) | 2–5 |
| 3 | 1.00 (1) | 0.00 (1) | 7 |
| 4 | 1.00 (1) | 0.00 (1) | 10 |

**The headline.** Before this family was rebuilt, `confinement` and
`noninterference` both failed at 0.00, *including compositionally with every
ancestor supplied*. After splitting the 37-line soundness proof into two
intermediate lemmas and dropping the target to 22 lines, the same policy at the
same one-shot budget **proves the full soundness theorem.** Nothing about the
theorem, language or model changed. Only the granularity of the trusted
interfaces.

## An accidental noise-floor measurement

Depth-1 rungs have no ancestors, so their monolithic and compositional prompts
are **byte-identical** (verified with `diff`). They are nonetheless answered by
different policy instances.

Compositional scored 8/8 on them, monolithic 6/8. So **run-to-run variance is
about 25%** at this sample size, the depth-1 gap above is noise rather than
signal, and every $n=1$ cell in this repository carries that variance.

Worth making deliberate: depth-1 rungs are a built-in replication control that
measures the noise floor for free.

## Evidence on depth versus volume

Within the compositional arm, where task length equals target length:

```
wt_anti     18 lines, depth 1  ->  proved
assign_ni    7 lines, depth 2  ->  failed
```

A longer proof succeeded while a shorter one failed, so length does not order
success. This is only visible because the rebuild deliberately added a
shallow-long rung and a deep-short rung, dropping corr(depth, LoC) from +0.97 to
+0.56.

The monolithic arm cuts the other way: totals of 13, 21, 25, 60 lines across
depths 2–4, all failing. With $n=1$ and 25% noise, unsettled.

\newpage

# Part VIII: What is known to be wrong or missing

* **Sample sizes.** Almost every interesting cell is $n=1$ against a 25% noise
  floor. Three anecdotes, not three rates.
* **Contamination is mitigated, not solved.** Renaming blocks verbatim recall of
  library proof terms, but a model that knows the Peano development or
  Volpano–Smith–Irvine can still transfer the strategy, and arguably should. The
  renaming defence is itself **untested**; comparing pass rates on renamed
  versus native `Nat` statements would measure it.
* **The causal question is open.** The chain/antichain contrast exists to settle
  it and has only been run against a policy that ceilings on it. Running it
  against Haiku, which fails in-band, is the cheapest remaining experiment.
* **Episodes are single-step.** One attempt, one whole proof, one binary reward.
  A per-tactic version with intermediate proof states as observations needs a
  persistent Lean server rather than one-shot compilation.
* **Scale.** The benchmark now has three theories and 41 rungs total. `lf-lean`
  operates on 1,276 statements. This is a demonstration of method, not a
  competing system.

# The finding worth keeping

Building this made one thing clear that arguing about it would not have.

The grader worked, the dependency graph worked, the renaming worked, the
certifications worked. The binding constraint was none of those. It was
**difficulty calibration.** Arithmetic ceilings at 1.00 for Sonnet; the original
noninterference family floors at 0.00 above depth 1; Haiku lands in between and
produces a curve. The informative band is narrow and neither family hit it by
accident.

A task that always passes and a task that always fails both yield zero gradient.
That is the same sparse-reward failure Theorem attribute to unit tests. So dense
signal is **not a property of proof assistants**. It is a property of a task
distribution matched to a policy. Proof assistants make dense signal possible,
not automatic, and anyone planning to scale RL on verification needs a
calibration story rather than a kernel.

\newpage

# Part IX: The full Vsi port

The benchmark's simplified noninterference family is not the security system
from the original OCaml repository. To close that gap, `lean/Vsi.lean` ports the
original system with its lattice, subtyping, nesting counters, while loops, and
function extension.

The final call extension adds three load-bearing pieces:

* `tyE_call_L` inverts a public call and forces both the declared argument and
  result levels to be public.
* `FTOk` states that every declared function signature is backed by a function
  table entry with matching parameter and return labels and a typed body.
* `agree` proves expression agreement and command noninterference together by
  induction on fuel. This is necessary because expressions can call commands,
  while commands evaluate expressions.

The final theorem is `noninterference_full`. Lean 4.32.2 compiles it, and its
`#print axioms` report is:

```
'noninterference_full' depends on axioms: [propext, Quot.sound]
```

There is no `sorryAx`. The result is termination-insensitive: both executions
are assumed to finish at the supplied fuel.

## Vsi benchmark family

The new `vsi` family extracts the actual declarations and reference proof blocks
from `lean/Vsi.lean` into 14 rungs over depths 1 through 6. Its dependency graph
includes the inversion lemmas, `secure_sub`, confinement, call-free expression
agreement, the combined `agree` theorem, and `noninterference_full`.

The family certification passed all three checks:

```
uv run python -m pdd.selftest --family vsi
family=vsi  14 rungs  depths 1..6  0 failure(s)
```

That certifies compositional solvability, monolithic solvability, and that each
declared dependency is required by the citing proof.

## Five-sample export

The requested export produced 420 prompts, with five samples per rung and three
seeds. `verify_groups` accepted seven isolated groups:

| group | prompts |
|---|---:|
| compositional depth 1 | 30 |
| compositional depth 2 | 30 |
| compositional depth 3 | 45 |
| compositional depth 4 | 60 |
| compositional depth 5 | 30 |
| compositional depth 6 | 15 |
| monolithic | 210 |

The plan was generated without merging groups. No policy subagent interface was
available in this run, so no `.out` responses were submitted. The report
therefore records every cell as `missing`, not as a model failure:

| depth | compositional | monolithic |
|---:|---:|---:|
| 1 | 0.00 (30 missing) | 0.00 (30 missing) |
| 2 | 0.00 (30 missing) | 0.00 (30 missing) |
| 3 | 0.00 (45 missing) | 0.00 (45 missing) |
| 4 | 0.00 (60 missing) | 0.00 (60 missing) |
| 5 | 0.00 (30 missing) | 0.00 (30 missing) |
| 6 | 0.00 (15 missing) | 0.00 (15 missing) |

These zeros must not be interpreted as pass rates. They are an empty policy
run, and no decay estimate can be inferred from them. A real sweep still needs
one independent policy instance per printed group, with no merging across
compositional depths.
