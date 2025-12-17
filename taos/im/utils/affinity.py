# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import os
import multiprocessing

def get_core_allocation():
    total_cores = multiprocessing.cpu_count()

    validator_count = max(2, int(total_cores * 0.30))
    query_count = max(2, int(total_cores * 0.25))
    reward_count = max(2, int(total_cores * 0.30))
    reporting_count = max(1, int(total_cores * 0.15)) 

    allocated = validator_count + query_count + reward_count + reporting_count
    if allocated > total_cores:
        scale = total_cores / allocated
        validator_count = max(2, int(validator_count * scale))
        query_count = max(2, int(query_count * scale))
        reward_count = max(2, int(reward_count * scale))
        reporting_count = max(1, int(reporting_count * scale))
        allocated = validator_count + query_count + reward_count + reporting_count

    return {
        'validator': list(range(0, validator_count)),
        'query': list(range(validator_count, validator_count + query_count)),
        'reward': list(range(validator_count + query_count, validator_count + query_count + reward_count)),
        'reporting': list(range(validator_count + query_count + reward_count, 
                               min(total_cores, validator_count + query_count + reward_count + reporting_count)))
    }