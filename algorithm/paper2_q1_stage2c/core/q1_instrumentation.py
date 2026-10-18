# -*- coding: utf-8 -*-
"""Instrumentation helpers for exact objective-evaluation counting."""

def install_evaluation_counter(GAcoreClass):
    if getattr(GAcoreClass, "_q1_counter_installed", False):
        return
    original = GAcoreClass.calculateFitnessObjectives
    def wrapped(self, chromosome):
        self.evaluation_count = getattr(self, "evaluation_count", 0) + 1
        return original(self, chromosome)
    GAcoreClass.calculateFitnessObjectives = wrapped
    GAcoreClass._q1_counter_installed = True
    GAcoreClass._q1_original_calculateFitnessObjectives = original
