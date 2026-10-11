/*
 * mcmf_sequential.c
 * Sequential implementation of Min-Cost Max-Flow using
 * Successive Shortest Paths with SPFA initial potentials and Priority Queue Dijkstra.
 * Standard C (C99).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#define INF 1000000000000000LL // 10^15
#define MAX_NODES 20000
#define MAX_EDGES 200000

typedef struct {
    int u;
    int v;
    long long cap;
    long long flow;
    long long cost;
    int rev; // index in adj[v]
} Edge;

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

// SPFA for initial potentials
int spfa(int s, long long *pi) {
    for (int i = 0; i < num_nodes; i++) pi[i] = INF;
    int *in_queue = (int*)calloc(num_nodes, sizeof(int));
    int *count = (int*)calloc(num_nodes, sizeof(int));
    int *queue = (int*)malloc(num_nodes * 20 * sizeof(int));

    int head = 0, tail = 0;
    pi[s] = 0;
    queue[tail++] = s;
    in_queue[s] = 1;
    count[s] = 1;

    while (head < tail) {
        int u = queue[head++];
        in_queue[u] = 0;

        for (int e = g.head[u]; e != -1; e = g.next[e]) {
            if (g.cap[e] - g.flow[e] > 0) {
                int v = g.to[e];
                if (pi[u] + g.cost[e] < pi[v]) {
                    pi[v] = pi[u] + g.cost[e];
                    if (!in_queue[v]) {
                        queue[tail++] = v;
                        in_queue[v] = 1;
                        count[v]++;
                        if (count[v] > num_nodes) {
                            // Negative cycle
                            free(in_queue); free(count); free(queue);
                            return 0;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        if (pi[i] >= INF / 2) pi[i] = 0;
    }

    free(in_queue);
    free(count);
    free(queue);
    return 1;
}

// Min-Heap for Dijkstra
typedef struct {
    long long dist;
    int u;
} HeapNode;

typedef struct {
    HeapNode *data;
    int size;
    int capacity;
} MinHeap;

MinHeap* create_heap(int cap) {
    MinHeap *h = (MinHeap*)malloc(sizeof(MinHeap));
    h->data = (HeapNode*)malloc(cap * sizeof(HeapNode));
    h->size = 0;
    h->capacity = cap;
    return h;
}

void heap_push(MinHeap *h, long long dist, int u) {
    int i = h->size++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->data[p].dist <= dist) break;
        h->data[i] = h->data[p];
        i = p;
    }
    h->data[i].dist = dist;
    h->data[i].u = u;
}

HeapNode heap_pop(MinHeap *h) {
    HeapNode res = h->data[0];
    HeapNode val = h->data[--h->size];
    int i = 0;
    while (i * 2 + 1 < h->size) {
        int a = i * 2 + 1, b = i * 2 + 2;
        int best = (b < h->size && h->data[b].dist < h->data[a].dist) ? b : a;
        if (h->data[best].dist >= val.dist) break;
        h->data[i] = h->data[best];
        i = best;
    }
    if (h->size > 0) h->data[i] = val;
    return res;
}

void free_heap(MinHeap *h) {
    free(h->data);
    free(h);
}

void solve_mcmf_sequential(int s, int t, long long max_flow_target, long long *out_flow, long long *out_cost) {
    long long *pi = (long long*)malloc(num_nodes * sizeof(long long));
    long long *dist = (long long*)malloc(num_nodes * sizeof(long long));
    int *parent_edge = (int*)malloc(num_nodes * sizeof(int));
    MinHeap *pq = create_heap(MAX_EDGES);

    if (!spfa(s, pi)) {
        printf("Error: Negative cost cycle detected!\n");
        return;
    }

    long long total_flow = 0;
    long long total_cost = 0;

    while (total_flow < max_flow_target) {
        for (int i = 0; i < num_nodes; i++) {
            dist[i] = INF;
            parent_edge[i] = -1;
        }

        dist[s] = 0;
        pq->size = 0;
        heap_push(pq, 0, s);

        while (pq->size > 0) {
            HeapNode top = heap_pop(pq);
            int u = top.u;
            long long d = top.dist;
            if (d > dist[u]) continue;

            for (int e = g.head[u]; e != -1; e = g.next[e]) {
                if (g.cap[e] - g.flow[e] > 0) {
                    int v = g.to[e];
                    long long reduced_cost = g.cost[e] + pi[u] - pi[v];
                    if (dist[u] + reduced_cost < dist[v]) {
                        dist[v] = dist[u] + reduced_cost;
                        parent_edge[v] = e;
                        heap_push(pq, dist[v], v);
                    }
                }
            }
        }

        if (dist[t] >= INF / 2) break; // No augmenting path

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
    free_heap(pq);
}

int main(int argc, char **argv) {
    // If graph file is supplied: ./mcmf_seq <graph_input_file>
    if (argc < 2) {
        // Small self-test
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
        solve_mcmf_sequential(0, 5, 7, &flow, &cost);
        printf("[SEQUENTIAL C] Test Flow: %lld, Min Cost: %lld\n", flow, cost);
        free_graph();
        return 0;
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

    clock_t t0 = clock();
    long long flow = 0, cost = 0;
    solve_mcmf_sequential(s, t, max_flow, &flow, &cost);
    clock_t t1 = clock();
    double time_ms = (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC;

    printf("{\"version\": \"Sequential\", \"flow\": %lld, \"cost\": %lld, \"passengers\": %lld, \"time_ms\": %.3f}\n",
           flow, cost, -cost, time_ms);

    free_graph();
    return 0;
}
