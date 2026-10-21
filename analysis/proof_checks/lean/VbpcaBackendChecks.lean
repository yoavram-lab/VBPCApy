/- Algebraic checks for exact backend rearrangements. These are finite-sum
identities, not a verification of inference, floating-point error or convergence. -/
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

set_option autoImplicit false
namespace VbpcaBackendChecks
open scoped BigOperators

/-- A complete rectangular sum factors without positivity or symmetry assumptions. -/
theorem rectangular_product {I J : Type*} [Fintype I] [Fintype J]
    (f : I → ℝ) (g : J → ℝ) :
    (∑ i, ∑ j, f i * g j) = (∑ i, f i) * (∑ j, g j) := by
  simp_rw [← Finset.mul_sum]
  exact (Finset.sum_mul _ _ _).symm

/-- For every component pair, the three covariance terms can be accumulated
from complete-data totals. Empty axes are allowed; no PSD hypothesis is used. -/
theorem covariance_component_pair {I J : Type*} [Fintype I] [Fintype J]
    (a b av : I → ℝ) (s t sv : J → ℝ) :
    (∑ i, ∑ j, (a i * sv j * b i + s j * av i * t j + av i * sv j)) =
      (∑ i, a i * b i) * (∑ j, sv j) +
      (∑ i, av i) * (∑ j, s j * t j) +
      (∑ i, av i) * (∑ j, sv j) := by
  have h : (∑ i, ∑ j, (a i * sv j * b i + s j * av i * t j + av i * sv j)) =
      ∑ i, ∑ j, ((a i * b i) * sv j + av i * (s j * t j) + av i * sv j) := by
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [h]
  simp only [Finset.sum_add_distrib]
  rw [rectangular_product, rectangular_product, rectangular_product]

/-- Observed plus missing partitions an arbitrary finite domain. Membership
is explicit; zeros are values and do not define observation status. -/
theorem observed_complete_minus_missing {I : Type*} (domain : Finset I)
    (observed : I → Prop) [DecidablePred observed] (f : I → ℝ) :
    (∑ i ∈ domain.filter observed, f i) = (∑ i ∈ domain, f i) -
      (∑ i ∈ domain.filter (fun i => ¬ observed i), f i) := by
  have h := Finset.sum_filter_add_sum_filter_not domain observed f
  linarith

/-- Flattened outer-product contractions retain all covariance cross terms.
No diagonal approximation or symmetry hypothesis is introduced. -/
theorem variance_entry_rearrangement {K : Type*} [Fintype K]
    (a s : K → ℝ) (av sv : K → K → ℝ) (muv : ℝ) :
    (∑ k, ∑ l, a k * sv k l * a l) +
      (∑ k, ∑ l, av k l * s k * s l) +
      (∑ k, ∑ l, av k l * sv k l) + muv =
    (∑ k, ∑ l, (a k * a l) * sv k l) +
      (∑ k, ∑ l, av k l * (s k * s l)) +
      (∑ k, ∑ l, av k l * sv k l) + muv := by
  have ha : (∑ k, ∑ l, a k * sv k l * a l) =
      ∑ k, ∑ l, (a k * a l) * sv k l := by
    apply Finset.sum_congr rfl
    intro k _
    apply Finset.sum_congr rfl
    intro l _
    ring
  have hs : (∑ k, ∑ l, av k l * s k * s l) =
      ∑ k, ∑ l, av k l * (s k * s l) := by
    apply Finset.sum_congr rfl
    intro k _
    apply Finset.sum_congr rfl
    intro l _
    ring
  rw [ha, hs]

/-- Finite summation axes may be permuted. No probabilistic integrability
or nonempty-dimension assumption is needed for these finite sums. -/
theorem component_sum_commute {I J K L : Type*}
    [Fintype I] [Fintype J] [Fintype K] [Fintype L] (f : I → J → K → L → ℝ) :
    (∑ i, ∑ j, ∑ k, ∑ l, f i j k l) = ∑ k, ∑ l, ∑ i, ∑ j, f i j k l := by
  calc
    _ = ∑ i, ∑ k, ∑ j, ∑ l, f i j k l := by
      apply Finset.sum_congr rfl
      intro i _
      exact Finset.sum_comm
    _ = ∑ k, ∑ i, ∑ j, ∑ l, f i j k l := Finset.sum_comm
    _ = ∑ k, ∑ i, ∑ l, ∑ j, f i j k l := by
      apply Finset.sum_congr rfl
      intro k _
      apply Finset.sum_congr rfl
      intro i _
      exact Finset.sum_comm
    _ = ∑ k, ∑ l, ∑ i, ∑ j, f i j k l := by
      apply Finset.sum_congr rfl
      intro k _
      exact Finset.sum_comm

/-- Full complete-data covariance contribution, including both quadratic
forms and the covariance cross term. The covariance arrays are unrestricted
real arrays here; Gaussian/mean-field assumptions belong to their statistical
interpretation, not to this algebraic identity. -/
theorem complete_covariance_contraction {I J K : Type*}
    [Fintype I] [Fintype J] [Fintype K]
    (a : I → K → ℝ) (s : K → J → ℝ)
    (av : I → K → K → ℝ) (sv : J → K → K → ℝ) :
    (∑ i, ∑ j, ∑ k, ∑ l,
      (a i k * sv j k l * a i l + s k j * av i k l * s l j + av i k l * sv j k l)) =
    ∑ k, ∑ l, ((∑ i, a i k * a i l) * (∑ j, sv j k l) +
      (∑ i, av i k l) * (∑ j, s k j * s l j) +
      (∑ i, av i k l) * (∑ j, sv j k l)) := by
  rw [component_sum_commute]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro l _
  exact covariance_component_pair (fun i => a i k) (fun i => a i l)
    (fun i => av i k l) (fun j => s k j) (fun j => s l j) (fun j => sv j k l)

#print axioms rectangular_product
#print axioms covariance_component_pair
#print axioms observed_complete_minus_missing
#print axioms variance_entry_rearrangement
#print axioms component_sum_commute
#print axioms complete_covariance_contraction
end VbpcaBackendChecks
