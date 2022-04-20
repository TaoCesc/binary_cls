def solution(jobs, index):
    time = 0
    last = 0
    for j in range(len(jobs)):
        min = 999
        min_i = 0
        for i, job in enumerate(jobs):
            if job == 0:
                continue
            if job < min:
                min_i = i
                min = job
            if job == min:
                continue
        if index == min_i:
            time += jobs[min_i]
            last = min_i
            jobs[min_i] = 0
            return time

        time += jobs[min_i]
        last = min_i
        jobs[min_i] = 0

print(solution([100],0))
