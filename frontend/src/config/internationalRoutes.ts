export interface GlobalWaypoint {
  id: string;
  label: string;
  coordinates: [number, number];
  description: string;
}

export type InternationalTransportMode = "ocean" | "air" | "overland" | "multimodal";

export interface InternationalRoute {
  origin: GlobalWaypoint;
  checkpoint: GlobalWaypoint;
  destination: GlobalWaypoint;
  mode: InternationalTransportMode;
  leg1Mode: Exclude<InternationalTransportMode, "multimodal">;
  leg2Mode: Exclude<InternationalTransportMode, "multimodal">;
}

export const ORIGIN_HUBS: GlobalWaypoint[] = [
  { id: "jebel-ali", label: "Dubai International Humanitarian City / Jebel Ali Port, UAE", coordinates: [55.027, 24.985], description: "Global staging and container relief hub" },
  { id: "dwc", label: "Al Maktoum International Cargo Hub (DWC), UAE", coordinates: [55.161, 24.896], description: "Strategic air cargo staging" },
  { id: "dxb", label: "Dubai International Airport Mega Terminal (DXB), UAE", coordinates: [55.365, 25.253], description: "Emergency airlift gateway" },
  { id: "salalah", label: "Port of Salalah Global Transshipment Hub, Oman", coordinates: [54.006, 16.935], description: "Indian Ocean transshipment hub" },
];

export const CHECKPOINT_HUBS: GlobalWaypoint[] = [
  { id: "djibouti", label: "Port of Djibouti / Doraleh Terminal, Djibouti", coordinates: [43.145, 11.595], description: "Horn of Africa Gateway" },
  { id: "jeddah", label: "King Abdulaziz Port / Jeddah Hub, Saudi Arabia", coordinates: [39.172, 21.485], description: "Red Sea Corridor" },
  { id: "mombasa", label: "Port of Mombasa Maritime Hub, Kenya", coordinates: [39.668, -4.043], description: "East Africa Corridor" },
  { id: "amman", label: "Amman / Queen Alia Regional Staging, Jordan", coordinates: [35.993, 31.722], description: "Levant Corridor" },
  { id: "el-arish", label: "El Arish Humanitarian Logistics Hub, Egypt", coordinates: [33.803, 31.132], description: "Sinai / Gaza Corridor" },
  { id: "karachi", label: "Port Qasim / Karachi Staging Hub, Pakistan", coordinates: [67.33, 24.785], description: "South Asia Corridor" },
];

export const DESTINATION_ZONES: GlobalWaypoint[] = [
  { id: "mogadishu", label: "Mogadishu Emergency Relief Zone, Somalia", coordinates: [45.318, 2.046], description: "Crisis response zone" },
  { id: "dadaab", label: "Nairobi / Dadaab Regional Relief Depot, Kenya", coordinates: [36.821, -1.292], description: "Regional relief depot" },
  { id: "port-sudan", label: "Port Sudan / Kassala Crisis Response, Sudan", coordinates: [37.216, 19.615], description: "Crisis response corridor" },
  { id: "aden", label: "Aden / Sana'a Humanitarian Corridor, Yemen", coordinates: [45.018, 12.785], description: "Humanitarian access corridor" },
  { id: "rafah", label: "Rafah / Gaza Humanitarian Crossing", coordinates: [34.259, 31.248], description: "Humanitarian crossing" },
  { id: "dhaka", label: "Dhaka / Cox's Bazar Relief Operations, Bangladesh", coordinates: [90.412, 23.81], description: "South Asia relief operations" },
  { id: "kathmandu", label: "Kathmandu Emergency Airlift Hub, Nepal", coordinates: [85.324, 27.717], description: "Emergency airlift hub" },
];

export const DEFAULT_INTERNATIONAL_ROUTE: InternationalRoute = {
  origin: ORIGIN_HUBS[0],
  checkpoint: CHECKPOINT_HUBS[0],
  destination: DESTINATION_ZONES[0],
  mode: "multimodal",
  leg1Mode: "ocean",
  leg2Mode: "overland",
};
