/-
Root module of the Lean companion to "Protein Interaction as an Operator".
One import per certified book example.
-/

import OperatorGraph.Anova
import OperatorGraph.Bayes
import OperatorGraph.NoRecovery
import OperatorGraph.PairLaw
import OperatorGraph.Rem
import OperatorGraph.Shadow

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
