export type VesselId = "LAD" | "LCX" | "RCA";

export type VesselSpec = {
  id: VesselId;
  targetKey: VesselId;
  label: string;
  description: string;
};

export const VESSELS: VesselSpec[] = [
  {
    id: "LAD",
    targetKey: "LAD",
    label: "Left Anterior Descending (LAD)",
    description: "Supplies the anterior interventricular septum and anterior wall.",
  },
  {
    id: "LCX",
    targetKey: "LCX",
    label: "Left Circumflex (LCX)",
    description: "Supplies lateral and posterior left ventricular segments.",
  },
  {
    id: "RCA",
    targetKey: "RCA",
    label: "Right Coronary Artery (RCA)",
    description: "Supplies the right atrium, RV, and inferior wall.",
  },
];
