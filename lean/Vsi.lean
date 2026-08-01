/- Port of ymiled/type-system-for-noninterference to Lean 4.
   Volpano-Smith-Irvine security types with subtyping, while loops, and the
   function extension. Phase 1: syntax, semantics, subtyping. -/

inductive Lvl where
  | L : Lvl
  | H : Lvl
  deriving DecidableEq, Repr

/-- `less_or_eq` from type_checker.ml: `_, H -> true`, reflexive otherwise. -/
def Lvl.le : Lvl → Lvl → Bool
  | _, .H => true
  | .L, .L => true
  | .H, .L => false

/-- Join. Used for expression levels. -/
def Lvl.join : Lvl → Lvl → Lvl
  | .H, _ => .H
  | _, .H => .H
  | .L, .L => .L

inductive Ty where
  | base (t : Lvl)              -- T τ
  | tvar (t : Lvl)              -- Var τ
  | cmd  (t1 t2 : Lvl)          -- Cmd(τ1, τ2)
  | ncmd (t : Lvl) (n : Nat)    -- Ncmd(τ, n)
  | func (t1 t2 : Lvl)          -- Func(τ1, τ2)
  deriving DecidableEq, Repr

inductive Exp where
  | int   (n : Nat)
  | evar  (x : Nat)
  | binop (a b : Exp)
  | call  (f : Nat) (a : Exp)
  deriving Repr

inductive Com where
  | skip   : Com
  | assign (x : Nat) (e : Exp)
  | seq    (c d : Com)
  | ite    (e : Exp) (c d : Com)
  | wh     (e : Exp) (c : Com)
  deriving Repr

/-- A function, in the shape `FuncDef` insists on: body is `c ; return y`. -/
structure Fn where
  param : Nat
  body  : Com
  retv  : Nat
  t1    : Lvl
  t2    : Lvl

abbrev St := Nat → Nat
abbrev Ctx := Nat → Ty
abbrev FTable := Nat → Option Fn

def upd (s : St) (x v : Nat) : St := fun y => if y = x then v else s y

-- Fuel-indexed semantics. `while` makes evaluation partial, so the theorem
-- this supports is termination-insensitive: it relates runs that both finish.
mutual
  def evalE (ft : FTable) : Nat → St → Exp → Option Nat
    | 0, _, _ => none
    | _+1, _, .int n => some n
    | _+1, s, .evar x => some (s x)
    | k+1, s, .binop a b => do
        let va ← evalE ft k s a
        let vb ← evalE ft k s b
        some (va + vb)
    | k+1, s, .call f a => do
        let fn ← ft f
        let va ← evalE ft k s a
        let s' ← evalC ft k (upd s fn.param va) fn.body
        some (s' fn.retv)

  def evalC (ft : FTable) : Nat → St → Com → Option St
    | 0, _, _ => none
    | _+1, s, .skip => some s
    | k+1, s, .assign x e => do
        let v ← evalE ft k s e
        some (upd s x v)
    | k+1, s, .seq c d => do
        let s' ← evalC ft k s c
        evalC ft k s' d
    | k+1, s, .ite e c d => do
        let v ← evalE ft k s e
        if v = 0 then evalC ft k s d else evalC ft k s c
    | k+1, s, .wh e c => do
        let v ← evalE ft k s e
        if v = 0 then some s
        else do
          let s' ← evalC ft k s c
          evalC ft k s' (.wh e c)
end

/-- Subtyping, transcribed from `check_sub_rules`. `cmd` is contravariant in its
    first component and covariant in its second. -/
inductive SubTy : Ty → Ty → Prop where
  | refl  (t : Ty) : SubTy t t
  | base  {a b : Lvl} : a.le b = true → SubTy (.base a) (.base b)
  | cmd   {a b a' b' : Lvl} :
      a'.le a = true → b.le b' = true → SubTy (.cmd a b) (.cmd a' b')
  | ncmd  {a a' : Lvl} {n : Nat} :
      a'.le a = true → SubTy (.ncmd a n) (.ncmd a' n)
  | ncmdCmd {a : Lvl} {n : Nat} : SubTy (.ncmd a n) (.cmd a .L)
  | funcNcmd {a b c : Lvl} {n : Nat} : SubTy (.func a b) (.ncmd c n)
  | func  {a b a' b' : Lvl} :
      a'.le a = true → b.le b' = true → SubTy (.func a b) (.func a' b')
  | trans {x y z : Ty} : SubTy x y → SubTy y z → SubTy x z

#print axioms evalE
#print axioms evalC

/-- Declared signatures for the function table: `f ↦ (τ1, τ2)`. -/
abbrev Sig := Nat → Option (Lvl × Lvl)

/-- Expression typing. Transcribed from `check_type`'s expression cases.
    Not mutual with `TyC`: expression typing never refers to command typing,
    and keeping them separate is what makes `induction` usable on `TyE`. -/
inductive TyE : Ctx → Sig → Exp → Ty → Prop where
  | evar {G : Ctx} {sg : Sig} {x : Nat} {t : Lvl} :
      G x = .tvar t → TyE G sg (.evar x) (.base t)
  | int {G : Ctx} {sg : Sig} {n : Nat} :
      TyE G sg (.int n) (.base .L)
  | binop {G : Ctx} {sg : Sig} {a b : Exp} {t : Lvl} :
      TyE G sg a (.base t) → TyE G sg b (.base t) →
      TyE G sg (.binop a b) (.base t)
    -- FuncNonVoidCallDeriv restricts the parameter level to L.
  | call {G : Ctx} {sg : Sig} {f : Nat} {e : Exp} {t2 : Lvl} :
      sg f = some (.L, t2) → TyE G sg e (.base .L) →
      TyE G sg (.call f e) (.base t2)
  | sub {G : Ctx} {sg : Sig} {e : Exp} {t t' : Ty} :
      TyE G sg e t → SubTy t t' → TyE G sg e t'

/-- Command typing. Transcribed from the command cases of `check_type`. -/
inductive TyC : Ctx → Sig → Com → Ty → Prop where
  | skip {G : Ctx} {sg : Sig} :
      TyC G sg .skip (.ncmd .H 1)
  | assign {G : Ctx} {sg : Sig} {x : Nat} {e : Exp} {t : Lvl} :
      G x = .tvar t → TyE G sg e (.base t) →
      TyC G sg (.assign x e) (.ncmd t 1)
    -- IfDeriv, Cmd(H,H): the guard may be secret.
  | iteHH {G : Ctx} {sg : Sig} {e : Exp} {c d : Com} :
      TyE G sg e (.base .H) →
      TyC G sg c (.cmd .H .H) → TyC G sg d (.cmd .H .H) →
      TyC G sg (.ite e c d) (.cmd .H .H)
    -- IfDeriv, general Cmd: the guard must be public.
  | iteL {G : Ctx} {sg : Sig} {e : Exp} {c d : Com} {a b : Lvl} :
      TyE G sg e (.base .L) →
      TyC G sg c (.cmd a b) → TyC G sg d (.cmd a b) →
      TyC G sg (.ite e c d) (.cmd a b)
    -- IfDeriv, Ncmd: nesting depth decreases in the branches.
  | iteN {G : Ctx} {sg : Sig} {e : Exp} {c d : Com} {t : Lvl} {n : Nat} :
      TyE G sg e (.base t) →
      TyC G sg c (.ncmd t n) → TyC G sg d (.ncmd t n) →
      TyC G sg (.ite e c d) (.ncmd t (n + 1))
  | whHH {G : Ctx} {sg : Sig} {e : Exp} {c : Com} :
      TyE G sg e (.base .H) → TyC G sg c (.cmd .H .H) →
      TyC G sg (.wh e c) (.cmd .H .H)
  | whL {G : Ctx} {sg : Sig} {e : Exp} {c : Com} {a b : Lvl} :
      TyE G sg e (.base .L) → TyC G sg c (.cmd a b) → a.le b = true →
      TyC G sg (.wh e c) (.cmd a b)
    -- SeqDeriv, Cmd(τ,H).
  | seqH {G : Ctx} {sg : Sig} {c d : Com} {t : Lvl} :
      TyC G sg c (.cmd t .H) → TyC G sg d (.cmd .H .H) →
      TyC G sg (.seq c d) (.cmd t .H)
    -- SeqDeriv, general.
  | seqL {G : Ctx} {sg : Sig} {c d : Com} {a b : Lvl} :
      TyC G sg c (.cmd a .L) → TyC G sg d (.cmd a b) →
      TyC G sg (.seq c d) (.cmd a b)
  | sub {G : Ctx} {sg : Sig} {c : Com} {t t' : Ty} :
      TyC G sg c t → SubTy t t' → TyC G sg c t'

/-- Public agreement: the attacker reads only variables typed `Var L`. -/
def lowEq (G : Ctx) (s t : St) : Prop := ∀ x, G x = .tvar .L → s x = t x

theorem lowEq_refl (G : Ctx) (s : St) : lowEq G s s := by
  intro x _; rfl

theorem lowEq_symm {G : Ctx} {s t : St} (h : lowEq G s t) : lowEq G t s := by
  intro x hx; exact (h x hx).symm

theorem lowEq_trans {G : Ctx} {s t u : St}
    (h1 : lowEq G s t) (h2 : lowEq G t u) : lowEq G s u := by
  intro x hx; exact (h1 x hx).trans (h2 x hx)

#print axioms lowEq_trans

theorem Lvl.le_refl (a : Lvl) : a.le a = true := by
  cases a <;> rfl

theorem Lvl.le_trans {a b c : Lvl} (h1 : a.le b = true) (h2 : b.le c = true) :
    a.le c = true := by
  cases a <;> cases b <;> cases c <;> simp_all [Lvl.le]

/-- Only a base type can subtype into a base type, and the levels are ordered.
    This is what makes the typing relation usable despite subsumption: without
    it, every inversion would have to consider `sub` as an open case. -/
theorem sub_base_inv {t : Ty} {b : Lvl} (h : SubTy t (.base b)) :
    ∃ a, t = .base a ∧ a.le b = true := by
  generalize hb : Ty.base b = tb at h
  induction h generalizing b with
  | refl u => exact ⟨b, hb.symm ▸ rfl, Lvl.le_refl b⟩
  | @base a b' hab => cases hb; exact ⟨a, rfl, hab⟩
  | cmd => cases hb
  | ncmd => cases hb
  | ncmdCmd => cases hb
  | funcNcmd => cases hb
  | func => cases hb
  | @trans x y z _ _ ihxy ihyz =>
    obtain ⟨a', hy, ha'⟩ := ihyz hb
    obtain ⟨a, hx, ha⟩ := ihxy hy.symm
    exact ⟨a, hx, Lvl.le_trans ha ha'⟩

#print axioms sub_base_inv

/-- Inversion at `base L`: subsumption cannot have widened the level, because
    `sub_base_inv` forces the source level `a` to satisfy `a.le L`, and only
    `L` does. This is the pattern every inversion below follows. -/
theorem tyE_var_L {G : Ctx} {sg : Sig} {x : Nat}
    (h : TyE G sg (.evar x) (.base .L)) : G x = .tvar .L := by
  generalize he : Exp.evar x = ex at h
  generalize ht : Ty.base Lvl.L = tl at h
  induction h with
  | evar hg => cases he; cases ht; exact hg
  | int => cases he
  | binop => cases he
  | call => cases he
  | sub _ hsub ih =>
    obtain ⟨a, hta, hle⟩ := sub_base_inv (ht ▸ hsub)
    cases a with
    | L => exact ih he hta.symm
    | H => simp [Lvl.le] at hle

#print axioms tyE_var_L

/-- Inversion for `binop` at `base L`. The typing rule types both operands at
    the *same* level, so a public sum has two public operands.

    Stated with the index equations as explicit hypotheses rather than via
    `generalize ... at h`. The latter also rewrites the goal, which breaks the
    `sub` case whenever the conclusion mentions the type being inverted. -/
theorem tyE_binop_L {G : Ctx} {sg : Sig} {a b : Exp}
    (h : TyE G sg (.binop a b) (.base .L)) :
    TyE G sg a (.base .L) ∧ TyE G sg b (.base .L) := by
  suffices H : ∀ (eb : Exp) (tl : Ty), TyE G sg eb tl →
      eb = .binop a b → tl = .base .L →
      TyE G sg a (.base .L) ∧ TyE G sg b (.base .L) from H _ _ h rfl rfl
  intro eb tl hh
  induction hh with
  | evar => intro heq _; cases heq
  | int => intro heq _; cases heq
  | binop ha hb => intro heq hteq; cases heq; cases hteq; exact ⟨ha, hb⟩
  | call => intro heq _; cases heq
  | sub _ hsub ih =>
    intro heq hteq
    subst hteq
    obtain ⟨c, hta, hle⟩ := sub_base_inv hsub
    cases c with
    | L => exact ih heq hta
    | H => simp [Lvl.le] at hle

/-- Expressions containing no function call. The call case of agreement needs
    the whole command-level argument, so it is separated out. -/
def noCall : Exp → Bool
  | .int _ => true
  | .evar _ => true
  | .binop a b => noCall a && noCall b
  | .call _ _ => false

/-- Public call-free expressions evaluate alike in agreeing states.
    Termination-insensitive: both runs are assumed to have produced a value at
    the same fuel. -/
theorem evalE_agree_nocall {G : Ctx} {sg : Sig} {ft : FTable} {s t : St}
    (hst : lowEq G s t) :
    ∀ (e : Exp), noCall e = true → TyE G sg e (.base .L) →
      ∀ (k : Nat) (v w : Nat),
        evalE ft k s e = some v → evalE ft k t e = some w → v = w := by
  intro e
  induction e with
  | int n =>
    intro _ _ k v w hv hw
    cases k with
    | zero => simp [evalE] at hv
    | succ m => simp [evalE] at hv hw; omega
  | evar x =>
    intro _ hty k v w hv hw
    cases k with
    | zero => simp [evalE] at hv
    | succ m =>
      simp [evalE] at hv hw
      have := hst x (tyE_var_L hty)
      omega
  | binop a b iha ihb =>
    intro hnc hty k v w hv hw
    cases k with
    | zero => simp [evalE] at hv
    | succ m =>
      simp [noCall, Bool.and_eq_true] at hnc
      obtain ⟨hta, htb⟩ := tyE_binop_L hty
      simp [evalE, Option.bind_eq_some_iff] at hv hw
      obtain ⟨va, hva, vb, hvb, hveq⟩ := hv
      obtain ⟨wa, hwa, wb, hwb, hweq⟩ := hw
      have e1 := iha hnc.1 hta m va wa hva hwa
      have e2 := ihb hnc.2 htb m vb wb hvb hwb
      omega
  | call f a _ => intro hnc; simp [noCall] at hnc

#print axioms evalE_agree_nocall

/-- "This type guarantees the command only writes secret variables."

    `func` is `True` only because `SubTy.funcNcmd` lets any function type
    subtype into any `ncmd`; no `TyC` rule ever produces a function type, so a
    command can never actually carry one. -/
def secure : Ty → Prop
  | .cmd a _ => a = .H
  | .ncmd a _ => a = .H
  | .func _ _ => True
  | _ => False

/-- Security is preserved *downwards* through subtyping, because `cmd` and
    `ncmd` are contravariant in the level that matters. -/
theorem secure_sub {t t' : Ty} (h : SubTy t t') (hs : secure t') : secure t := by
  induction h with
  | refl => exact hs
  | base => exact hs.elim
  | @cmd a b a' b' ha _ =>
    simp only [secure] at hs ⊢
    subst hs; cases a <;> simp [Lvl.le] at ha ⊢
  | @ncmd a a' n ha =>
    simp only [secure] at hs ⊢
    subst hs; cases a <;> simp [Lvl.le] at ha ⊢
  | @ncmdCmd a n => simp only [secure] at hs ⊢; exact hs
  | funcNcmd => trivial
  | func => trivial
  | trans _ _ ih1 ih2 => exact ih1 (ih2 hs)

/-- **Confinement.** A command whose type guarantees it writes only secret
    variables leaves the public projection of the state unchanged.

    Induction is on *fuel* first and the typing derivation second. The `while`
    case is why: at fuel `k+1` it runs the body at fuel `k` and then the loop
    again at fuel `k`, so the induction hypothesis must range over all commands
    at smaller fuel, which induction on the derivation alone does not give. -/
theorem confinement {G : Ctx} {sg : Sig} {ft : FTable} :
    ∀ (k : Nat) (c : Com) (ty : Ty), TyC G sg c ty → secure ty →
      ∀ (s s' : St), evalC ft k s c = some s' → lowEq G s s' := by
  intro k
  induction k with
  | zero => intro c ty _ _ s s' hev; simp [evalC] at hev
  | succ m ih =>
    intro c ty hty
    induction hty with
    | skip => intro _ s s' hev; simp [evalC] at hev; subst hev; exact lowEq_refl G s
    | @assign x e t hx _ =>
      intro hs s s' hev
      simp only [secure] at hs; subst hs
      simp [evalC, Option.bind_eq_some_iff] at hev
      obtain ⟨v, _, hs'⟩ := hev
      subst hs'
      intro y hy
      have hyx : y ≠ x := by
        intro hcon; rw [hcon, hx] at hy; exact absurd hy (by simp)
      simp [upd, hyx]
    | @iteHH e c d _ _ _ ihc ihd =>
      intro hs s s' hev
      simp [evalC, Option.bind_eq_some_iff] at hev
      obtain ⟨v, _, hbr⟩ := hev
      by_cases hz : v = 0
      · simp [hz] at hbr; exact ih d _ (by assumption) hs s s' hbr
      · simp [hz] at hbr; exact ih c _ (by assumption) hs s s' hbr
    | @iteL e c d a b _ _ _ ihc ihd =>
      intro hs s s' hev
      simp [evalC, Option.bind_eq_some_iff] at hev
      obtain ⟨v, _, hbr⟩ := hev
      by_cases hz : v = 0
      · simp [hz] at hbr; exact ih d _ (by assumption) hs s s' hbr
      · simp [hz] at hbr; exact ih c _ (by assumption) hs s s' hbr
    | @iteN e c d t n _ _ _ ihc ihd =>
      intro hs s s' hev
      simp only [secure] at hs; subst hs
      simp [evalC, Option.bind_eq_some_iff] at hev
      obtain ⟨v, _, hbr⟩ := hev
      by_cases hz : v = 0
      · simp [hz] at hbr; exact ih d _ (by assumption) (by simp [secure]) s s' hbr
      · simp [hz] at hbr; exact ih c _ (by assumption) (by simp [secure]) s s' hbr
    | @whHH e c he hc _ =>
      intro hs s s' hev
      simp [evalC, Option.bind_eq_some_iff] at hev
      obtain ⟨v, _, hbr⟩ := hev
      by_cases hz : v = 0
      · simp [hz] at hbr; subst hbr; exact lowEq_refl G s
      · simp [hz, Option.bind_eq_some_iff] at hbr
        obtain ⟨u, hu, hloop⟩ := hbr
        -- The loop's own derivation is rebuilt, since induction on the
        -- derivation consumed it and the recursive call needs it back.
        exact lowEq_trans (ih c _ hc hs s u hu)
          (ih (.wh e c) _ (TyC.whHH he hc) hs u s' hloop)
    | @whL e c a b he hc hle _ =>
      intro hs s s' hev
      simp [evalC, Option.bind_eq_some_iff] at hev
      obtain ⟨v, _, hbr⟩ := hev
      by_cases hz : v = 0
      · simp [hz] at hbr; subst hbr; exact lowEq_refl G s
      · simp [hz, Option.bind_eq_some_iff] at hbr
        obtain ⟨u, hu, hloop⟩ := hbr
        exact lowEq_trans (ih c _ hc hs s u hu)
          (ih (.wh e c) _ (TyC.whL he hc hle) hs u s' hloop)
    | @seqH c d t _ _ ihc ihd =>
      intro hs s s' hev
      simp [evalC, Option.bind_eq_some_iff] at hev
      obtain ⟨u, hu, hd⟩ := hev
      exact lowEq_trans (ih c _ (by assumption) hs s u hu)
        (ih d _ (by assumption) (by simp [secure]) u s' hd)
    | @seqL c d a b _ _ ihc ihd =>
      intro hs s s' hev
      simp [evalC, Option.bind_eq_some_iff] at hev
      obtain ⟨u, hu, hd⟩ := hev
      exact lowEq_trans (ih c _ (by assumption) (by simp only [secure] at hs ⊢; exact hs) s u hu)
        (ih d _ (by assumption) hs u s' hd)
    | @sub c t t' _ hsub ihc =>
      intro hs s s' hev
      exact ihc (secure_sub hsub hs) s s' hev

#print axioms confinement
