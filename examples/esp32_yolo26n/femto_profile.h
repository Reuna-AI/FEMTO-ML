/*
 * femto_profile.h -- where run_model()'s time goes, node by node.
 *
 * Built with -D FEMTO_PROFILE=1, the generated run_model() calls femto_prof_node()
 * before each node and femto_prof_end() after the last; nothing is regenerated.
 * femto_prof_report() then writes, for the runs since the last report: the time per
 * run, the share of each op type, and the slowest nodes.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef FEMTO_PROFILE_H
#define FEMTO_PROFILE_H

#ifdef __cplusplus
extern "C" {
#endif

void femto_prof_node(int index, const char *op, const char *name);
void femto_prof_end(void);

/* The report as text into [out] (at most [size] bytes, NUL-terminated), and the
 * counts reset. Returns the length written. */
int femto_prof_report(char *out, int size);

#ifdef __cplusplus
}
#endif

#endif
