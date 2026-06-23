#  $(INF_TEMPLATE)
#         | <--------------------- clean-infer 
#         | (macro expansion)            ^
#         V                              |
#  $(INF_SCRIPT)                         |
#         | (resource calculation)       |
#         V                              |
#  $(INF_SBATCH)                         |
#         | (resource calculation)       |
#         \_-----------------------------/


#  $(CNT_TEMPLATE)
#         | <--------------------- clean-infer 
#         | (macro expansion)            ^
#         V                              |
#  $(CNT_SCRIPT)                         |
#         |------------------------------/
#


#  $(MET_TEMPLATE)
#         | <--------------------- clean-infer 
#         | (macro expansion)            ^
#         V                              |
#  $(MET_SCRIPT)                         |
#         | (resource calculation)       |
#         V                              |
#  $(MET_SBATCH)                         |
#         | (resource calculation)       |
#         \------------------------------/


mk-slurm: $(INF_SCRIPT) $(CNT_SCRIPT) $(MET_SCRIPT) $(INF_CALLS) $(SACRE_CALLS) $(INF_SBATCH) $(MET_SBATCH)
> @echo "mk-slurm.mk: ✅ Slurm planning complete."; \
> echo "mk-slurm.mk:     - bin dir: $(SELFDIR)"; \
> echo "mk-slurm.mk:     - account: $(JOBPROJECT)"; \
> echo "mk-slurm.mk:     - outdir: $(OUTDIR)"; \
> echo "mk-slurm.mk:     - Inferences:"; \
> echo "mk-slurm.mk:         calls: $(INF_CALLS)"; \
> echo "mk-slurm.mk:         template: $(INF_TEMPLATE)"; \
> echo "mk-slurm.mk:         script: $(INF_SCRIPT)"; \
> echo "mk-slurm.mk:         sbatch: $(INF_SBATCH)"; \
> echo "mk-slurm.mk:     - Continuation:"; \
> echo "mk-slurm.mk:         template: $(CNT_TEMPLATE)"; \
> echo "mk-slurm.mk:         script: $(CNT_SCRIPT)"; \
> echo "mk-slurm.mk:     - Scoring:"; \
> echo "mk-slurm.mk:         calls: $(SACRE_CALLS)"; \
> echo "mk-slurm.mk:         template: $(MET_TEMPLATE)"; \
> echo "mk-slurm.mk:         script: $(MET_SCRIPT)"; \
> echo "mk-slurm.mk:         sbatch: $(MET_SBATCH)"; \
> echo "mk-slurm.mk: ✨ I am happy with slurm run preparations."; \
> echo "mk-slurm.mk: ❓ Human approval required before submission:"; \
> cat "$(INF_SBATCH)" | sed 's/^/mk-slurm.mk:      /'; \
> cat "$(MET_SBATCH)" | sed 's/^/mk-slurm.mk:      /'; \
> echo "mk-slurm.mk:    Run: make run-infer"; \
> echo

# creates the model specific slurm script for inferences
$(INF_SCRIPT): $(INF_TEMPLATE) $(SELFDIR)/mk-infer.mk mk-calls clean-lock 
> @ncalls="$$(wc -l < $(OUTDIR)/calls.out)"; \
> echo "mk-slurm.mk: ✅ Inference slurm template found $<"; \
> echo "mk-slurm.mk:    Creating a script for job $(FIRST_GOAL)_inf_$${ncalls}_tasks..."; \
> m4 \
>   -D__SIF__="$(SIF)" \
>   -D__ACCOUNT__="$(JOBPROJECT)" \
>   -D__DISKPROJECT__="$(DISKPROJECT)" \
>   -D__SELFDIR__="$(SELFDIR)" \
>   -D__OUTDIR__="$(OUTDIR)" \
>   -D__ACTIVATE__="$(INF_ACTIVATE)" \
>   -D__MAKESCRIPT__="$(INF_SBATCH)" \
>   -D__LOGDIR__="$(LOGDIR)" \
>   -D__JOB_NAME__="$(FIRST_GOAL)_inf_$${ncalls}_tasks" \
>   -D__MODELDIR__="$(MODELDIR)" \
>   -D__FLAG__="$(INF_FLAG)" \
>   -D__DONE__="$(INF_DONE)" \
>   "$<" >"$@"; \
> echo "mk-slurm.mk: ✅ Job-specific inference slurm script now at:";\
> echo "mk-slurm.mk:    $@"; \
> chmod +x "$@"

# creates the model specific slurm script for continuation
$(CNT_SCRIPT): $(CNT_TEMPLATE) $(SELFDIR)/mk-infer.mk mk-calls clean-lock 
> @set -euo pipefail; \
> echo "mk-slurm.mk: ✅ Continuation slurm template found $<"; \
> echo "mk-slurm.mk:    Creating a script for job $(FIRST_GOAL)_cnt..."; \
> m4 \
>   -D__ACCOUNT__="$(JOBPROJECT)" \
>   -D__DISKPROJECT__="$(DISKPROJECT)" \
>   -D__SELFDIR__="$(SELFDIR)" \
>   -D__OUTDIR__="$(OUTDIR)" \
>   -D__MAKESCRIPT__="$(INF_SBATCH)" \
>   -D__LOGDIR__="$(LOGDIR)" \
>   -D__JOB_NAME__="$(FIRST_GOAL)_cnt" \
>   -D__MODELDIR__="$(MODELDIR)" \
>   -D__FLAG__="$(CNT_FLAG)" \
>   -D__DONE__="$(CNT_DONE)" \
>   "$<" >"$@"; 
> @chmod +x "$@"
> @echo "mk-slurm.mk: ✅ Job-specific continuation slurm script now at:";\
> @echo "mk-slurm.mk:    $@"; \
> chmod +x "$@"

# creates the model specific slurm script for scoring
$(MET_SCRIPT): $(MET_TEMPLATE) $(SELFDIR)/mk-score.mk mk-calls clean-lock 
> @set -euo pipefail; \
> @ncalls="$$(wc -l < $(OUTDIR)/calls.sacre.out)"; \
> echo "mk-slurm.mk: ✅ Scoring slurm template found $<"; \
> echo "mk-slurm.mk:    Creating a script for job $(FIRST_GOAL)_met_$${ncalls}_tasks..."; \
> m4 \
>   -D__ACCOUNT__="$(JOBPROJECT)" \
>   -D__DISKPROJECT__="$(DISKPROJECT)" \
>   -D__SELFDIR__="$(SELFDIR)" \
>   -D__OUTDIR__="$(OUTDIR)" \
>   -D__MAKESCRIPT__="$(MET_SBATCH)" \
>   -D__LOGDIR__="$(LOGDIR)" \
>   -D__JOB_NAME__="$(FIRST_GOAL)_met_$${ncalls}_tasks" \
>   -D__ACTIVATE__="$(SACRE_ACTIVATE)" \
>   -D__MODELDIR__="$(MODELDIR)" \
>   -D__FLAG__="$(MET_FLAG)" \
>   -D__DONE__="$(MET_DONE)" \
>   "$<" >"$@"; \
> echo "mk-slurm.mk: ✅ Job-specific scoring slurm script now at:";\
> echo "mk-slurm.mk:    $@"; \
> chmod +x "$@"

$(INF_SBATCH): $(INF_CALLS) $(INF_SCRIPT) mk-calls clean-lock 
> @bash $(INF_SCRIPT); \
> echo "mk-slurm.mk: ✅ Created the sbatch command:"; \
> cat $(INF_SBATCH) | sed 's/^/mk-slurm.mk:    /'

$(MET_SBATCH): $(SACRE_CALLS) $(MET_SCRIPT) mk-calls clean-lock 
> @bash $(MET_SCRIPT); \
> echo "mk-slurm.mk: ✅ Created the sbatch command:"; \
> cat $(MET_SBATCH) | sed 's/^/mk-slurm.mk:    /'

slurm: $(INF_SCRIPT) $(INF_BATCH) $(CNT_SCRIPT) $(MET_SCRIPT) $(MET_BATCH)
> @true

clean-slurm:
> rm -f $(INF_SCRIPT) $(INF_BATCH) $(CNT_SCRIPT) $(MET_SCRIPT) $(MET_BATCH)

