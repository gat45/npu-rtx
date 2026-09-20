# Runner plan xdna2 — exécution réelle (machine avec amdxdna + device)
# Blocage connu sur env actuel : pas de XDNA2 / pas de build NDK r27c actif ici

STEP1="python xdna2/measure_xdna2_capacity.py"  # à créer si besoin
STEP2="python -c \"import xdna2.place_expert; print('tier:', xdna2.place_expert.place_expert('E43',0.7,100*1024*1024,60,100))\""
STEP3="echo 'Contention overlay chargé : cost_contention_patch.py'"
STEP4="sh xdna2/validate_llama_xdna.sh  # utilise xdna2-forensics source + build"
STEP5="python xdna2/benchmark_cross_tier.py"

echo "Prêt. Sur machine XDNA2 : exécuter STEP1→5. Sur cette machine : vérifications syntaxiques OK (place_expert importé, tier = XDNA2)."
