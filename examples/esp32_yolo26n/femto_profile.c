/*
 * femto_profile.c -- see femto_profile.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "femto_profile.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(ESP_PLATFORM)
#  include "esp_timer.h"
static int64_t now_us(void) { return esp_timer_get_time(); }
#else
#  include <time.h>
static int64_t now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000 + t.tv_nsec / 1000;
}
#endif

#define MAX_NODES 512
#define MAX_TYPES 48
#define TOP_NODES 12

static int32_t s_node_us[MAX_NODES];   /* microseconds: 35 minutes before they wrap */
static const char *s_node_op[MAX_NODES];
static const char *s_node_name[MAX_NODES];
static int s_nodes = 0;
static int s_runs = 0;
static int s_current = -1;
static int64_t s_started = 0;

static void close_current(int64_t t)
{
    if (s_current >= 0 && s_current < MAX_NODES) s_node_us[s_current] += (int32_t)(t - s_started);
    s_current = -1;
}

void femto_prof_node(int index, const char *op, const char *name)
{
    int64_t t = now_us();
    close_current(t);
    if (index < 0 || index >= MAX_NODES) return;
    s_node_op[index] = op;
    s_node_name[index] = name;
    if (index + 1 > s_nodes) s_nodes = index + 1;
    s_current = index;
    s_started = now_us();
}

void femto_prof_end(void)
{
    close_current(now_us());
    s_runs++;
}

int femto_prof_report(char *out, int size)
{
    int len = 0;
#define PUT(...) do { if (len < size) len += snprintf(out + len, size - len, __VA_ARGS__); } while (0)

    if (!s_runs) { PUT("femto profile: no runs yet\n"); return len < size ? len : size - 1; }

    int64_t total = 0;
    const char *type_name[MAX_TYPES];
    int64_t type_us[MAX_TYPES];
    int type_count[MAX_TYPES];
    int types = 0;

    for (int i = 0; i < s_nodes; i++) {
        if (!s_node_op[i]) continue;
        total += s_node_us[i];
        int t = 0;
        while (t < types && strcmp(type_name[t], s_node_op[i]) != 0) t++;
        if (t == types && types < MAX_TYPES) { type_name[t] = s_node_op[i]; type_us[t] = 0; type_count[t] = 0; types++; }
        if (t < types) { type_us[t] += s_node_us[i]; type_count[t]++; }
    }
    if (total <= 0) total = 1;

    PUT("femto profile: %d run%s, %lld ms a run\n", s_runs, s_runs == 1 ? "" : "s",
        (long long)(total / s_runs / 1000));

    /* op types, slowest first */
    for (int shown = 0; shown < types; shown++) {
        int best = -1;
        for (int t = 0; t < types; t++)
            if (type_count[t] >= 0 && (best < 0 || type_us[t] > type_us[best])) best = t;
        if (best < 0) break;
        PUT("  %-16s %3d nodes %7lld ms %5.1f%%\n", type_name[best], type_count[best],
            (long long)(type_us[best] / s_runs / 1000), 100.0 * type_us[best] / total);
        type_count[best] = -1;
    }

    /* slowest nodes */
    PUT("  slowest nodes:\n");
    static unsigned char taken[MAX_NODES];
    memset(taken, 0, sizeof(taken));
    for (int k = 0; k < TOP_NODES; k++) {
        int best = -1;
        for (int i = 0; i < s_nodes; i++)
            if (s_node_op[i] && !taken[i] && (best < 0 || s_node_us[i] > s_node_us[best])) best = i;
        if (best < 0) break;
        taken[best] = 1;
        PUT("  %6lld ms  %-12s %s\n", (long long)(s_node_us[best] / s_runs / 1000),
            s_node_op[best], s_node_name[best]);
    }

    memset(s_node_us, 0, sizeof(s_node_us));
    s_runs = 0;
#undef PUT
    return len < size ? len : size - 1;
}
