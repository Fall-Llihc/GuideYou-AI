// Konstanta titik mulai (home) dan kategori destinasi.
//
// MUST stay in sync dengan notebook §2.6 (HOME_OPTIONS) dan §2.1 (CATEGORY_ORDER).
// Sinkronisasi tersebut penting karena RL agent dilatih dengan home points ini —
// memilih home di luar daftar masih jalan, tapi distribusi reward bisa shift.

// 5 home presets resmi — sama persis dengan notebook §2.6
export const HOME_OPTIONS = [
  { id: "alun-alun",   name: "Alun-Alun Bandung", lat: -6.9215, lng: 107.6071 },
  { id: "stasiun",     name: "Stasiun Bandung",   lat: -6.9145, lng: 107.6020 },
  { id: "lembang",     name: "Pasar Lembang",     lat: -6.8126, lng: 107.6178 },
  { id: "dago",        name: "Dago",              lat: -6.8915, lng: 107.6107 },
  { id: "gedung-sate", name: "Gedung Sate",       lat: -6.9024, lng: 107.6188 },
];

// Hanya 3 kategori sesuai notebook v3:
//   - Alam     : gunung, curug, kebun teh, danau, tebing, hutan, taman
//   - Kuliner  : restoran, kafe, warung, food court, kaki lima
//   - Wisata   : theme park, water park, zoo, agro, hot spring,
//                mall, factory outlet, pasar (Wisata Umum termasuk shopping)
//
// "Belanja" digabung ke "Wisata"; "Budaya" dihapus karena banyak data noise
// (masjid/sekolah/kantor) di OSM tag-nya.
export const ALL_CATEGORIES = ["Alam", "Kuliner", "Wisata"];

// Optional: deskripsi pendek untuk tooltip di chip kategori
export const CATEGORY_DESCRIPTIONS = {
  Alam:    "Gunung, curug, kebun teh, taman alam",
  Kuliner: "Restoran, kafe, warung, food court",
  Wisata:  "Theme park, agro, factory outlet, mall, hot spring",
};
