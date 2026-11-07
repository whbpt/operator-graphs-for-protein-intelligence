/-
Root module of the Lean companion to "Protein Interaction as an Operator".
One import per certified book example.
-/

import OperatorGraph.Anova
import OperatorGraph.Bayes
import OperatorGraph.Epistasis
import OperatorGraph.MeanSpectrum
import OperatorGraph.NoRecovery
import OperatorGraph.PairLaw
import OperatorGraph.Rem
import OperatorGraph.Shadow
import OperatorGraph.ThreeState
import OperatorGraph.TwoBackground

export OperatorGraph.PairLaw (diagonalCoupling antidiagonalCoupling diagonalCoupling_normalized
  antidiagonalCoupling_normalized no_joint_law no_joint_law_marginals no_joint_law_measure)

export OperatorGraph.Anova (walsh_orthonormal walsh_coefficients meanH_G residual_G
  dynamic_variance inner_one_sjk correlated_cross)

export OperatorGraph.NoRecovery (entropy_processed_le push_comp no_recovery)

export OperatorGraph.Shadow (GA Bm GB fro_GA fro_GB shadow_eq shadow_GA
  row_means_GA col_means_GA row_means_GB col_means_GB
  spec_GA_1 spec_GA_2 spec_GA_ker spec_Bm_1 spec_Bm_2 spec_Bm_ker
  witness_det action_GA action_Bm action_GB action_diff)

export OperatorGraph.Bayes (bayes_floor_bound randomized_convex)

export OperatorGraph.Rem (order_profile_sum)

export OperatorGraph.ThreeState (decompose_T decompose_matrix action_f)

export OperatorGraph.MeanSpectrum (LA_eig_0_a LA_eig_0_b LA_eig_2
  LB_eig_0_a LB_eig_0_b LB_eig_2
  Lbar_eig_0 Lbar_eig_half Lbar_eig_three_halves
  witness_independent LA_zero_modes_independent)

export OperatorGraph.Epistasis (additive_null log_coordinate_id multiplicative_null)

export OperatorGraph.TwoBackground (fro_G0 fro_H inner_G0_H dynamic_norm)
