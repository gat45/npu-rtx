#!/usr/bin/env python3
"""residency_manager.py - ExpertResidencyManager (trou n°24).

Gerer la residence des experts : SSD / RAM / PINNED / VRAM / XDNA-local.
API : promote(), evict(), get(), pin_shared(). Etats et generations (DynaExQ-style).
Le runtime reel (pread/cudaMemcpyAsync/kernel) sera branche par les executors.
"""

import json
import os
import time


class ExpertResidencyManager:
    def __init__(self, max_vram_gib=6.5):
        self._slots = {}          # key -> {"location","bytes","generation","last_use"}
        self.max_vram_gib = max_vram_gib
        self.vram_used = 0.0

    def _key(self, layer, expert, tensor="gate"):
        return f"{layer:02d}/{expert:03d}/{tensor}"

    def get(self, layer, expert, tensor="gate"):
        k = self._key(layer, expert, tensor)
        e = self._slots.get(k)
        if e:
            e["last_use"] = time.time()
            return e
        return None

    def promote(self, layer, expert, tensor, location, bytes_, precision):
        """Deplacer un expert vers une localisation. Retourne True si ok."""
        k = self._key(layer, expert, tensor)
        if location == "VRAM":
            new_usage = self.vram_used + bytes_ / (1024**3)
            if new_usage > self.max_vram_gib:
                return False  # contrainte dure VRAM
            self.vram_used = new_usage
        # si deja en VRAM et on en sort, liberer
        old = self._slots.get(k)
        if old and old["location"] == "VRAM":
            self.vram_used -= old["bytes"] / (1024**3)
        self._slots[k] = {"location": location, "bytes": bytes_,
                          "precision": precision, "generation": (old or {}).get("generation", 0) + 1,
                          "last_use": time.time()}
        return True

    def pin_shared(self, layer, expert, tensor, bytes_):
        """Shared expert : residency ALWAYS (jamais evince)."""
        k = self._key(layer, expert, tensor)
        self._slots[k] = {"location": "VRAM", "bytes": bytes_, "precision": "BF16",
                          "generation": 0, "last_use": time.time(), "pinned": True}
        self.vram_used += bytes_ / (1024**3)

    def evict(self, victim_key):
        e = self._slots.pop(victim_key, None)
        if e and e["location"] == "VRAM" and not e.get("pinned"):
            self.vram_used -= e["bytes"] / (1024**3)

    def stats(self):
        return {"slots": len(self._slots), "vram_gib": round(self.vram_used, 3),
                "max_vram_gib": self.max_vram_gib}


if __name__ == "__main__":
    rm = ExpertResidencyManager()
    print("promote E17->VRAM:", rm.promote(0, 17, "gate", "VRAM", 1_769_472, "Q4"))
    print("promote E92->RAM:", rm.promote(0, 92, "up", "RAM", 1_769_472, "Q4"))
    rm.pin_shared(0, 0, "shared", 6_291_456)
    print("stats:", rm.stats())
    print("get E17:", bool(rm.get(0, 17, "gate")))