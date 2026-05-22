// Itinerary planner — nearest-neighbor TSP from home through chosen destinations
window.buildItinerary = function(params) {
  const { home, startMin, endMin, maxKm, count, categories, budget } = params;
  const all = window.DESTINATIONS;

  // Filter by categories if any selected
  let pool = categories.length > 0
    ? all.filter(d => categories.includes(d.category))
    : all.slice();

  // Score destinations by rating + slight randomness so it feels personalized
  pool.forEach(d => { d._score = d.rating * 10 + Math.random() * 3; });
  pool.sort((a, b) => b._score - a._score);

  // Build chain: nearest-neighbor from home, with constraints
  const chain = [];
  let cursor = home;
  let used = new Set();
  let spent = 0;
  // Take top portion of pool by score, then nearest-neighbor walk
  const topPool = pool.slice(0, Math.max(count + 4, 8));

  for (let i = 0; i < count; i++) {
    let best = null, bestDist = Infinity;
    for (const d of topPool) {
      if (used.has(d.id)) continue;
      const dist = window.distKm(cursor, d);
      if (maxKm && dist > maxKm) continue;
      if (budget != null && spent + d.ticket > budget) continue;
      if (dist < bestDist) { bestDist = dist; best = d; }
    }
    if (!best) {
      // relax distance/budget constraints
      for (const d of topPool) {
        if (used.has(d.id)) continue;
        const dist = window.distKm(cursor, d);
        if (dist < bestDist) { bestDist = dist; best = d; }
      }
    }
    if (!best) break;
    used.add(best.id);
    chain.push({ ...best, distFromPrev: bestDist });
    spent += best.ticket;
    cursor = best;
  }

  // Compute schedule
  // average travel speed in city: 28 km/h
  const SPEED = 28;
  const steps = [];
  let t = startMin;
  let cursorPt = home;
  let totalCost = 0;
  let totalKm = 0;

  for (let i = 0; i < chain.length; i++) {
    const d = chain[i];
    const travelMin = Math.round((d.distFromPrev / SPEED) * 60);
    const arriveAt = t + travelMin;
    const departAt = arriveAt + d.duration;
    steps.push({
      idx: i + 1,
      dest: d,
      travelMin,
      travelKm: d.distFromPrev,
      arriveAt,
      departAt,
    });
    totalCost += d.ticket;
    totalKm += d.distFromPrev;
    t = departAt;
    cursorPt = d;
  }

  // Return trip
  const returnKm = chain.length > 0 ? window.distKm(cursorPt, home) : 0;
  const returnMin = Math.round((returnKm / SPEED) * 60);
  totalKm += returnKm;

  const totalTime = t + returnMin - startMin;
  const arriveHome = t + returnMin;

  return {
    steps,
    totalCost,
    totalKm,
    totalTime,
    returnKm,
    returnMin,
    arriveHome,
    overBudget: arriveHome > endMin,
    spareMin: endMin - arriveHome,
  };
};

// Generate a narrative story for the itinerary
window.generateNarrative = function(itin, params) {
  if (itin.steps.length === 0) return "Belum ada destinasi yang dipilih.";

  const homeName = params.homeName;
  const first = itin.steps[0].dest;
  const last = itin.steps[itin.steps.length - 1].dest;
  const cats = [...new Set(itin.steps.map(s => s.dest.category))];

  const vibePhrases = {
    "Alam": "menghirup udara segar pegunungan",
    "Kuliner": "memanjakan lidah dengan kuliner khas",
    "Budaya": "menyelami warisan budaya Sunda",
    "Wisata": "menikmati hiburan dan rekreasi",
    "Belanja": "berburu oleh-oleh dan fashion lokal",
  };
  const vibes = cats.map(c => vibePhrases[c]).join(", lalu ");

  const intro = `Hari ini perjalananmu dimulai dari **${homeName}** menuju petualangan yang dirancang khusus buat kamu. Dengan ${itin.steps.length} destinasi terpilih, kamu akan ${vibes}. Sebuah rute yang ${itin.totalKm < 50 ? "ringkas dan padat" : "cukup berkelana"} — total ${itin.totalKm.toFixed(1)} km dalam waktu ${Math.floor(itin.totalTime/60)} jam ${itin.totalTime % 60} menit.`;

  const highlights = itin.steps.map(s => {
    const arr = window.fmtTime(s.arriveAt);
    return `**${s.dest.name}** *(${s.dest.category})* — tiba sekitar pukul ${arr.replace(".", ":")}, alokasikan ${s.dest.duration} menit untuk menyerap suasananya. ${s.dest.desc}.`;
  });

  const tips = [];
  if (first.category === "Alam") tips.push("Datang lebih pagi biar view ${first.name} masih clear dan nggak terlalu ramai.".replace("${first.name}", first.name));
  if (cats.includes("Kuliner")) tips.push("Sisakan ruang di perut — Bandung punya cara unik buat bikin kamu nagih.");
  if (itin.totalKm > 80) tips.push("Total jarak " + itin.totalKm.toFixed(1) + " km — sewa mobil/motor atau pakai Grab biar hemat waktu.");
  if (itin.spareMin > 60) tips.push("Masih ada " + Math.floor(itin.spareMin/60) + " jam ekstra — bisa mampir ke kafe di Braga sebelum pulang.");
  if (last.category === "Kuliner" || last.category === "Belanja") tips.push("Penutup di " + last.name + " pas banget — santai sambil bawa oleh-oleh.");
  if (tips.length === 0) tips.push("Bawa power bank dan air minum yang cukup — Bandung itu indah, tapi cuacanya unpredictable.");

  const closing = `Udah, gausah mikir panjang — **save itinerary ini** dan langsung cus! Bandung lagi nunggu kamu di ujung ${last.name}.`;

  return {
    intro,
    highlights,
    tips,
    closing,
    vibe: cats.length === 1 ? cats[0] : "Campuran",
  };
};
