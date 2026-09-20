#!/usr/bin/env python3
"""online_calibration.py - boucle d'apprentissage D2 (trou n°19).

prediction -> actual -> error -> confidence -> correction.
Conserve l'historique (plan_id, token, layer, expert). Chaque prediction porte
mean/variance/samples/confidence. Le planner ne doit pas choisir un candidat a
forte variance juste parce que la moyenne est plus basse.
"""

import json
import os
import statistics
import time


class OnlineCalibration:
    def __init__(self, path=None):
        self.path = path or os.path.join(os.path.dirname(__file__), "..", "runs",
                                         "calibration.jsonl")
        self.history = []

    def record(self, plan_id, key, predicted, actual):
        err = actual - predicted
        rec = {"ts": time.time(), "plan_id": plan_id, "key": key,
               "predicted": predicted, "actual": actual, "error": err}
        self.history.append(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return err

    def model_for(self, key):
        """Retourne (mean_pred, variance, samples, confidence) pour une cle."""
        vals = [h for h in self.history if h["key"] == key]
        if len(vals) < 3:
            return None  # pas assez de donnees -> pas de decision fiable
        preds = [v["predicted"] for v in vals]
        mean = statistics.mean(preds)
        variance = statistics.pvariance(preds) if len(preds) > 1 else 0.0
        std = variance ** 0.5
        # confidence : decroit avec la variance relative
        conf = max(0.0, 1.0 - (std / (abs(mean) + 1e-9)))
        return {"mean": mean, "variance": variance, "std": std,
                "samples": len(vals), "confidence": round(conf, 3)}


if __name__ == "__main__":
    c = OnlineCalibration()
    # simule : prediction H2D E17 (80us) vs actual (95us)
    c.record("plan-C1", "h2d:0/17/gate", 80, 95)
    c.record("plan-C1", "h2d:0/17/gate", 82, 97)
    c.record("plan-C1", "h2d:0/17/gate", 79, 88)
    print("modele h2d:0/17/gate :", c.model_for("h2d:0/17/gate"))
    print("NB : le planner NE doit PAS choisir un candidat a forte variance")
    print("juste parce que la moyenne est plus basse (regle confidence).")