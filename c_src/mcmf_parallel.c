/*
 * mcmf_parallel.c
 * Parallel OpenMP implementation of Min-Cost Max-Flow.
 * Leverages multi-threading using OpenMP for:
 * 1. Parallel SPFA edge-relaxation passes / distance checks.
 * 2. Parallel Dijkstra node updates and potential maintenance (Crauser et al. / Δ-stepping relaxation style).
 * 3. Parallel potential shifts across all vertices.
 *
 * Compiles with: gcc -O3 -fopenmp mcmf_parallel.c -o mcmf_parallel.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#define INF 1000000000000000LL // 10^15
#define MAX_NODES 20000
#define MAX_EDGES 200000

typedef struct {
    int *head;
    int *to;
    long long *cap;
    long long *flow;
    long long *cost;
    int *rev;
    int *next;
    int edge_cnt;
} Graph;

Graph g;
int num_nodes;

void init_graph(int n) {
    num_nodes = n;
    g.edge_cnt = 0;
    g.head = (int*)malloc(n * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) g.head[i] = -1;

    g.to = (int*)malloc(MAX_EDGES * sizeof(int));
    g.cap = (long long*)malloc(MAX_EDGES * sizeof(long long));
    g.flow = (long long*)malloc(MAX_EDGES * sizeof(long long));
    g.cost = (long long*)malloc(MAX_EDGES * sizeof(long long));
    g.rev = (int*)malloc(MAX_EDGES * sizeof(int));
    g.next = (int*)malloc(MAX_EDGES * sizeof(int));
}

void free_graph() {
    free(g.head);
    free(g.to);
    free(g.cap);
    free(g.flow);
    free(g.cost);
    free(g.rev);
    free(g.next);
}

void add_edge(int u, int v, long long cap, long long cost) {
    int e1 = g.edge_cnt++;
    int e2 = g.edge_cnt++;

    g.to[e1] = v; g.cap[e1] = cap; g.flow[e1] = 0; g.cost[e1] = cost; g.rev[e1] = e2;
    g.next[e1] = g.head[u]; g.head[u] = e1;

    g.to[e2] = u; g.cap[e2] = 0; g.flow[e2] = 0; g.cost[e2] = -cost; g.rev[e2] = e1;
    g.next[e2] = g.head[v]; g.head[v] = e2;
}

// Parallel SPFA / Bellman-Ford using OpenMP across active frontier vertices
int spfa_parallel(int s, long long *pi) {
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) pi[i] = INF;
    
    int *in_queue = (int*)calloc(num_nodes, sizeof(int));
    int *count = (int*)calloc(num_nodes, sizeof(int));
    int *curr_q = (int*)malloc(num_nodes * sizeof(int));
    int *next_q = (int*)malloc(num_nodes * sizeof(int));

    pi[s] = 0;
    curr_q[0] = s;
    int curr_size = 1;
    in_queue[s] = 1;
    count[s] = 1;

    int iter = 0;
    while (curr_size > 0 && iter <= num_nodes) {
        iter++;
        int next_size = 0;

        #pragma omp parallel
        {
            int local_next[1024];
            int local_cnt = 0;

            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < curr_size; i++) {
                int u = curr_q[i];
                in_queue[u] = 0;

                for (int e = g.head[u]; e != -1; e = g.next[e]) {
                    if (g.cap[e] - g.flow[e] > 0) {
                        int v = g.to[e];
                        long long new_d = pi[u] + g.cost[e];
                        if (new_d < pi[v]) {
                            #pragma omp critical
                            {
                                if (new_d < pi[v]) {
                                    pi[v] = new_d;
                                    if (!in_queue[v]) {
                                        in_queue[v] = 1;
                                        if (local_cnt < 1024) {
                                            local_next[local_cnt++] = v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            {
                for (int j = 0; j < local_cnt; j++) {
                    if (next_size < num_nodes) {
                        next_q[next_size++] = local_next[j];
                    }
                }
            }
        }

        // Swap queues
        int *tmp = curr_q;
        curr_q = next_q;
        next_q = tmp;
        curr_size = next_size;
    }

    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        if (pi[i] >= INF / 2) pi[i] = 0;
    }

    free(in_queue);
    free(count);
    free(curr_q);
    free(next_q);
    return 1;
}

// Parallel Dijkstra using work-sharing and concurrent relaxation
void solve_mcmf_parallel(int s, int t, long long max_flow_target, long long *out_flow, long long *out_cost, int num_threads) {
    omp_set_num_threads(num_threads);

    long long *pi = (long long*)malloc(num_nodes * sizeof(long long));
    long long *dist = (long long*)malloc(num_nodes * sizeof(long long));
    int *parent_edge = (int*)malloc(num_nodes * sizeof(int));
    int *visited = (int*)malloc(num_nodes * sizeof(int));

    spfa_parallel(s, pi);

    long long total_flow = 0;
    long long total_cost = 0;

    while (total_flow < max_flow_target) {
        #pragma omp parallel for
        for (int i = 0; i < num_nodes; i++) {
            dist[i] = INF;
            parent_edge[i] = -1;
            visited[i] = 0;
        }

        dist[s] = 0;

        // Dijkstra search loop
        for (int step = 0; step < num_nodes; step++) {
            int u = -1;
            long long min_d = INF;

            // Find min-distance unvisited node in parallel
            #pragma omp parallel
            {
                int local_u = -1;
                long long local_min = INF;

                #pragma omp for nowait
                for (int i = 0; i < num_nodes; i++) {
                    if (!visited[i] && dist[i] < local_min) {
                        local_min = dist[i];
                        local_u = i;
                    }
                }

                #pragma omp critical
                {
                    if (local_min < min_d) {
                        min_d = local_min;
                        u = local_u;
                    }
                }
            }

            if (u == -1 || min_d >= INF / 2) break;
            visited[u] = 1;

            // Relax outgoing edges from u
            for (int e = g.head[u]; e != -1; e = g.next[e]) {
                if (g.cap[e] - g.flow[e] > 0) {
                    int v = g.to[e];
                    long long reduced_cost = g.cost[e] + pi[u] - pi[v];
                    if (dist[u] + reduced_cost < dist[v]) {
                        dist[v] = dist[u] + reduced_cost;
                        parent_edge[v] = e;
                    }
                }
            }
        }

        if (dist[t] >= INF / 2) break; // No augmenting path found

        // Update potentials in parallel
        #pragma omp parallel for
        for (int i = 0; i < num_nodes; i++) {
            if (dist[i] < INF / 2) {
                pi[i] += dist[i];
            }
        }

        long long push = max_flow_target - total_flow;
        int curr = t;
        while (curr != s) {
            int e = parent_edge[curr];
            long long rem = g.cap[e] - g.flow[e];
            if (rem < push) push = rem;
            int rev_e = g.rev[e];
            curr = g.to[rev_e];
        }

        if (push <= 0) break;

        curr = t;
        long long path_cost = 0;
        while (curr != s) {
            int e = parent_edge[curr];
            g.flow[e] += push;
            g.flow[g.rev[e]] -= push;
            path_cost += g.cost[e];
            int rev_e = g.rev[e];
            curr = g.to[rev_e];
        }

        total_flow += push;
        total_cost += push * path_cost;
    }

    *out_flow = total_flow;
    *out_cost = total_cost;

    free(pi);
    free(dist);
    free(parent_edge);
    free(visited);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        // Self-test
        init_graph(6);
        add_edge(0, 1, 5, -20);
        add_edge(0, 2, 4, -15);
        add_edge(1, 3, 3, -10);
        add_edge(1, 4, 4, -5);
        add_edge(2, 3, 2, -18);
        add_edge(2, 4, 3, -8);
        add_edge(3, 5, 6, -2);
        add_edge(4, 5, 5, -1);

        long long flow = 0, cost = 0;
        solve_mcmf_parallel(0, 5, 7, &flow, &cost, 4);
        printf("[PARALLEL OpenMP C] Test Flow: %lld, Min Cost: %lld\n", flow, cost);
        free_graph();
        return 0;
    }

    int threads = 4;
    if (argc >= 3) {
        threads = atoi(argv[2]);
        if (threads <= 0) threads = 4;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "Cannot open graph file %s\n", argv[1]);
        return 1;
    }

    int n, m, s, t;
    long long max_flow;
    if (fscanf(f, "%d %d %d %d %lld", &n, &m, &s, &t, &max_flow) != 5) {
        fprintf(stderr, "Invalid graph header format\n");
        fclose(f);
        return 1;
    }

    init_graph(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        long long cap, cost;
        if (fscanf(f, "%d %d %lld %lld", &u, &v, &cap, &cost) == 4) {
            add_edge(u, v, cap, cost);
        }
    }
    fclose(f);

    double t0 = omp_get_wtime();
    long long flow = 0, cost = 0;
    solve_mcmf_parallel(s, t, max_flow, &flow, &cost, threads);
    double t1 = omp_get_wtime();
    double time_ms = (t1 - t0) * 1000.0;

    printf("{\"version\": \"Parallel_OpenMP\", \"threads\": %d, \"flow\": %lld, \"cost\": %lld, \"passengers\": %lld, \"time_ms\": %.3f}\n",
           threads, flow, cost, -cost, time_ms);

    free_graph();
    return 0;
}
