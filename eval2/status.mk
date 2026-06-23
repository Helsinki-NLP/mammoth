> @set -euo pipefail; \
> printf '%-22s %-5s %-6s %-10s %-5s %-5s %-6s %-6s %-6s %-6s\n' "model" "yamls" "calls" "HH:MMxGPUs" "elaps" "hyp" "0shyp" "calls" "sacre"; \
> printf '%-22s %-5s %-6s %-10s %-5s %-5s %-6s %-6s %-6s %-6s\n' "----------------------" "-----" "------" "----------" "-----" "-----" "------" "------" "------"; \
> for a in $(MODEL_ALIASES); do \
>   eval model_dir=\"\$${MODEL_$$a}\"; \
>   yaml_ok='0'; \
>   calls='0'; \
>   infer_info=''; \
>   runtime=''; \
>   done='0'; \
>   hyp_count='0'; \
>   zhyp_count='0'; \
>   sacre='0'; \
>   comet='0'; \
>   err_file="$$model_dir/testing.yaml.err"; \
>   yaml_dir="$$model_dir/inf_out"; \
>   if [ -d "$$yaml_dir" ]; then \
>     yaml_ok="$$(find "$$yaml_dir" -maxdepth 1 -type f -name '*.yaml' | wc -l)"; \
>   fi; \
>   calls_file="$$model_dir/inf_out/calls.out"; \
>   if [ -f "$$calls_file" ]; then \
>     calls="$$(wc -l < "$$calls_file")"; \
>   fi; \
>   inf_slurm="$$model_dir/inf_out/inf.slurm"; inf_sbatch="$$model_dir/inf_out/inf.sbatch"; \
>   if [ -s "$$inf_sbatch" ]; then \
>     line="$$(cat "$$inf_sbatch")"; \
>     t="$$(printf '%s\n' "$$line" | sed -n 's/.*--time=\([^ ]*\).*/\1/p')"; \
>     n="$$(printf '%s\n' "$$line" | sed -n 's/.*--nodes=\([0-9][0-9]*\).*/\1/p')"; \
>     gpn="$$(printf '%s\n' "$$line" | sed -n 's/.*--gpus-per-node=\([0-9][0-9]*\).*/\1/p')"; \
>     g="$$(printf '%s\n' "$$line" | sed -n 's/.*--gpus=\([0-9][0-9]*\).*/\1/p')"; \
>     ta="$$(printf '%s\n' "$$line" | sed -n 's/.*--ntasks=\([0-9][0-9]*\).*/\1/p')"; \
>     if [ -n "$$n" ] && [ -n "$$gpn" ]; then \
>       total_gpus=$$(( n * gpn )); \
>     elif [ -n "$$g" ]; then \
>       total_gpus="$$g"; \
>     elif [ -n "$$ta" ]; then \
>       total_gpus="$$ta"; \
>     else \
>       total_gpus="?"; \
>     fi; \
>     if [ -n "$$t" ]; then \
>       case "$$t" in \
>         *-*) \
>           d="$${t%%-*}"; \
>           rest="$${t#*-}"; \
>           hh="$${rest%%:*}"; \
>           rest="$${rest#*:}"; \
>           mm="$${rest%%:*}"; \
>           hh=$$((10#$$d * 24 + 10#$$hh)); \
>           mm=$$((10#$$mm)); \
>           ;; \
>         *) \
>           hh="$${t%%:*}"; \
>           rest="$${t#*:}"; \
>           mm="$${rest%%:*}"; \
>           hh=$$((10#$$hh)); \
>           mm=$$((10#$$mm)); \
>           ;; \
>       esac; \
>       pretty_t="$$(printf '%02d:%02d' "$$hh" "$$mm")"; \
>       infer_info="$$pretty_t"x"$$total_gpus"; \
>     else \
>       infer_info="[x]"; \
>     fi; \
>   else \
>     infer_info=''; \
>   fi; \
>   inf_done="$$model_dir/inference.done"; \
>   if [ -z "$$runtime" ] && [ -e "$$inf_done" ]; then \
>     runtime=' DONE'; \
>   fi; \
>   inf_flag="$$model_dir/inference.submitted"; \
>   if [ -f "$$inf_flag" ]; then \
>     jobid="$$(cat "$$inf_flag" 2>/dev/null || true)"; \
>     if [ -n "$$jobid" ]; then \
>       runtime_raw="$$(squeue -h -j "$$jobid" -o '%M' 2>/dev/null | head -1 || true)"; \
>       if [ -n "$$runtime_raw" ]; then \
>         case "$$runtime_raw" in \
>           *-*) \
>             d="$${runtime_raw%%-*}"; \
>             rest="$${runtime_raw#*-}"; \
>             hh="$${rest%%:*}"; \
>             mm="$${rest#*:}"; \
>             mm="$${mm%%:*}"; \
>             runtime="$$(printf '%02d:%02d' $$((10#$$d * 24 + 10#$$hh)) $$((10#$$mm)))"; \
>             ;; \
>           *:*:*) \
>             hh="$${runtime_raw%%:*}"; \
>             rest="$${runtime_raw#*:}"; \
>             mm="$${rest%%:*}"; \
>             runtime="$$(printf '%02d:%02d' $$((10#$$hh)) $$((10#$$mm)))"; \
>             ;; \
>           *:*) \
>             mm="$${runtime_raw%%:*}"; \
>             ss="$${runtime_raw#*:}"; \
>             runtime="$$(printf '00:%02d' $$((10#$$mm)))"; \
>             ;; \
>           *) \
>             runtime="$$runtime_raw"; \
>             ;; \
>         esac; \
>       else \
>         runtime=''; \
>       fi; \
>       if [ -z "$$runtime" ]; then \
>         rm -f "$$inf_flag"; \
>       fi; \
>     fi; \
>   fi; \
>   sacre_calls=''; \
>   if [ -d "$$model_dir/inf_out" ]; then \
>     hyp_count="$$(find "$$model_dir/inf_out" -maxdepth 1 -type f -name '*.hyp' | wc -l)"; \
>     zhyp_count="$$(find "$$model_dir/inf_out" -maxdepth 1 -type f -name '*.0shyp' | wc -l)"; \
>     sacre_count="$$(find "$$model_dir/inf_scores" -maxdepth 1 -type f -name '*.sacre' | wc -l 2>/dev/null || true)"; \
>     sacre="$$sacre_count"; \
>     sacre_calls_file="$$model_dir/inf_out/calls.sacre.out"; \
>     if [ -f "$$sacre_calls_file" ]; then sacre_calls="$$(grep -cw 'sacrebleu' "$$sacre_calls_file" || true)"; fi; \
>     comet_file="$$model_dir/inf_out/calls.comet.out"; \
>     if [ -f "$$comet_file" ]; then comet="$$(wc -l < "$$comet_file")"; fi; \
>   fi; \
>   if [ "$$zhyp_count" -gt 0 ]; then \
>     done="$$hyp_count+$$zhyp_count"; \
>   else \
>     done="$$hyp_count"; \
>   fi; \
>   printf '%-22s %-5s %-6s %-10s %-5s %-5s %-6s %-6s %-6s\n' "$$a" "$$yaml_ok" "$$calls" "$$infer_info" "$$runtime" "$$hyp_count" "$$zhyp_count" "$$sacre_calls" "$$sacre"; \
> done; \
> printf '%-22s %-5s %-6s %-10s %-5s %-5s %-6s %-6s %-6s %-6s\n' "----------------------" "-----" "------" "----------" "-----" "-----" "------" "------" "------"; \

