// Demo scenarios for the UI, mirroring the official test cases so the
// reviewer can reproduce every behavior from the interface in one click.

export const SCENARIOS = [
  {
    id: "clean_approval",
    label: "Clean consultation — full approval (TC004)",
    payload: {
      member_id: "EMP001", policy_id: "PLUM_GHI_2024",
      claim_category: "CONSULTATION", treatment_date: "2024-11-01",
      claimed_amount: 1500, ytd_claims_amount: 5000,
      documents: [
        { file_id: "F007", actual_type: "PRESCRIPTION", content: {
          doctor_name: "Dr. Arun Sharma", doctor_registration: "KA/45678/2015",
          patient_name: "Rajesh Kumar", date: "2024-11-01",
          diagnosis: "Viral Fever",
          medicines: ["Paracetamol 650mg", "Vitamin C 500mg"] } },
        { file_id: "F008", actual_type: "HOSPITAL_BILL", content: {
          hospital_name: "City Clinic, Bengaluru", patient_name: "Rajesh Kumar",
          date: "2024-11-01",
          line_items: [
            { description: "Consultation Fee", amount: 1000 },
            { description: "CBC Test", amount: 300 },
            { description: "Dengue NS1 Test", amount: 200 } ],
          total: 1500 } },
      ],
    },
  },
  {
    id: "network_discount",
    label: "Network hospital — discount before co-pay (TC010)",
    payload: {
      member_id: "EMP010", policy_id: "PLUM_GHI_2024",
      claim_category: "CONSULTATION", treatment_date: "2024-11-03",
      claimed_amount: 4500, hospital_name: "Apollo Hospitals",
      ytd_claims_amount: 8000,
      documents: [
        { file_id: "F019", actual_type: "PRESCRIPTION", content: {
          doctor_name: "Dr. S. Iyer", doctor_registration: "TN/56789/2013",
          patient_name: "Deepak Shah", diagnosis: "Acute Bronchitis",
          medicines: ["Amoxicillin 500mg", "Salbutamol Inhaler"] } },
        { file_id: "F020", actual_type: "HOSPITAL_BILL", content: {
          hospital_name: "Apollo Hospitals", patient_name: "Deepak Shah",
          line_items: [
            { description: "Consultation Fee", amount: 1500 },
            { description: "Medicines", amount: 3000 } ],
          total: 4500 } },
      ],
    },
  },
  {
    id: "dental_partial",
    label: "Dental — partial, cosmetic excluded (TC006)",
    payload: {
      member_id: "EMP002", policy_id: "PLUM_GHI_2024",
      claim_category: "DENTAL", treatment_date: "2024-10-15",
      claimed_amount: 12000,
      documents: [
        { file_id: "F011", actual_type: "HOSPITAL_BILL", content: {
          hospital_name: "Smile Dental Clinic", patient_name: "Priya Singh",
          line_items: [
            { description: "Root Canal Treatment", amount: 8000 },
            { description: "Teeth Whitening", amount: 4000 } ],
          total: 12000 } },
      ],
    },
  },
  {
    id: "waiting_period",
    label: "Diabetes inside waiting period — rejected (TC005)",
    payload: {
      member_id: "EMP005", policy_id: "PLUM_GHI_2024",
      claim_category: "CONSULTATION", treatment_date: "2024-10-15",
      claimed_amount: 3000,
      documents: [
        { file_id: "F009", actual_type: "PRESCRIPTION", content: {
          doctor_name: "Dr. Sunil Mehta", doctor_registration: "GJ/56789/2014",
          patient_name: "Vikram Joshi", diagnosis: "Type 2 Diabetes Mellitus",
          medicines: ["Metformin 500mg", "Glimepiride 1mg"] } },
        { file_id: "F010", actual_type: "HOSPITAL_BILL", content: {
          patient_name: "Vikram Joshi", date: "2024-10-15", total: 3000 } },
      ],
    },
  },
  {
    id: "unreadable_document",
    label: "Unreadable document — re-upload requested (TC002)",
    payload: {
      member_id: "EMP004", policy_id: "PLUM_GHI_2024",
      claim_category: "PHARMACY", treatment_date: "2024-10-25",
      claimed_amount: 800,
      documents: [
        { file_id: "F003", file_name: "prescription.jpg",
          actual_type: "PRESCRIPTION", quality: "GOOD" },
        { file_id: "F004", file_name: "blurry_bill.jpg",
          actual_type: "PHARMACY_BILL", quality: "UNREADABLE" },
      ],
    },
  },
  {
    id: "pre_auth_missing",
    label: "MRI without pre-authorization — rejected (TC007)",
    payload: {
      member_id: "EMP007", policy_id: "PLUM_GHI_2024",
      claim_category: "DIAGNOSTIC", treatment_date: "2024-11-02",
      claimed_amount: 15000,
      documents: [
        { file_id: "F012", actual_type: "PRESCRIPTION", content: {
          doctor_name: "Dr. Venkat Rao", doctor_registration: "AP/67890/2017",
          diagnosis: "Suspected Lumbar Disc Herniation",
          tests_ordered: ["MRI Lumbar Spine"] } },
        { file_id: "F013", actual_type: "LAB_REPORT", content: {
          test_name: "MRI Lumbar Spine" } },
        { file_id: "F014", actual_type: "HOSPITAL_BILL", content: {
          line_items: [{ description: "MRI Lumbar Spine", amount: 15000 }],
          total: 15000 } },
      ],
    },
  },
  {
    id: "per_claim_limit",
    label: "Per-claim limit exceeded — rejected (TC008)",
    payload: {
      member_id: "EMP003", policy_id: "PLUM_GHI_2024",
      claim_category: "CONSULTATION", treatment_date: "2024-10-20",
      claimed_amount: 7500, ytd_claims_amount: 10000,
      documents: [
        { file_id: "F015", actual_type: "PRESCRIPTION", content: {
          doctor_name: "Dr. R. Gupta", doctor_registration: "DL/34567/2016",
          diagnosis: "Gastroenteritis",
          medicines: ["Antibiotics", "Probiotics", "ORS"] } },
        { file_id: "F016", actual_type: "HOSPITAL_BILL", content: {
          line_items: [
            { description: "Consultation Fee", amount: 2000 },
            { description: "Medicines", amount: 5500 } ],
          total: 7500 } },
      ],
    },
  },
  {
    id: "wrong_document",
    label: "Wrong document type — stopped early (TC001)",
    payload: {
      member_id: "EMP001", policy_id: "PLUM_GHI_2024",
      claim_category: "CONSULTATION", treatment_date: "2024-11-01",
      claimed_amount: 1500,
      documents: [
        { file_id: "F001", file_name: "dr_sharma_prescription.jpg", actual_type: "PRESCRIPTION" },
        { file_id: "F002", file_name: "another_prescription.jpg", actual_type: "PRESCRIPTION" },
      ],
    },
  },
  {
    id: "different_patients",
    label: "Documents for different patients — stopped early (TC003)",
    payload: {
      member_id: "EMP001", policy_id: "PLUM_GHI_2024",
      claim_category: "CONSULTATION", treatment_date: "2024-11-01",
      claimed_amount: 1500,
      documents: [
        { file_id: "F005", file_name: "prescription_rajesh.jpg",
          actual_type: "PRESCRIPTION", patient_name_on_doc: "Rajesh Kumar" },
        { file_id: "F006", file_name: "bill_arjun.jpg",
          actual_type: "HOSPITAL_BILL", patient_name_on_doc: "Arjun Mehta" },
      ],
    },
  },
  {
    id: "fraud_same_day",
    label: "4th same-day claim — manual review (TC009)",
    payload: {
      member_id: "EMP008", policy_id: "PLUM_GHI_2024",
      claim_category: "CONSULTATION", treatment_date: "2024-10-30",
      claimed_amount: 4800,
      claims_history: [
        { claim_id: "CLM_0081", date: "2024-10-30", amount: 1200, provider: "City Clinic A" },
        { claim_id: "CLM_0082", date: "2024-10-30", amount: 1800, provider: "City Clinic B" },
        { claim_id: "CLM_0083", date: "2024-10-30", amount: 2100, provider: "Wellness Center" },
      ],
      documents: [
        { file_id: "F017", actual_type: "PRESCRIPTION", content: {
          diagnosis: "Migraine", doctor_name: "Dr. S. Khan" } },
        { file_id: "F018", actual_type: "HOSPITAL_BILL", content: { total: 4800 } },
      ],
    },
  },
  {
    id: "component_failure",
    label: "Component failure — graceful degradation (TC011)",
    payload: {
      member_id: "EMP006", policy_id: "PLUM_GHI_2024",
      claim_category: "ALTERNATIVE_MEDICINE", treatment_date: "2024-10-28",
      claimed_amount: 4000, simulate_component_failure: true,
      documents: [
        { file_id: "F021", actual_type: "PRESCRIPTION", content: {
          doctor_name: "Vaidya T. Krishnan", doctor_registration: "AYUR/KL/2345/2019",
          diagnosis: "Chronic Joint Pain", treatment: "Panchakarma Therapy" } },
        { file_id: "F022", actual_type: "HOSPITAL_BILL", content: {
          hospital_name: "Ayur Wellness Centre", total: 4000,
          line_items: [
            { description: "Panchakarma Therapy (5 sessions)", amount: 3000 },
            { description: "Consultation", amount: 1000 } ] } },
      ],
    },
  },
  {
    id: "excluded",
    label: "Excluded obesity treatment — rejected (TC012)",
    payload: {
      member_id: "EMP009", policy_id: "PLUM_GHI_2024",
      claim_category: "CONSULTATION", treatment_date: "2024-10-18",
      claimed_amount: 8000,
      documents: [
        { file_id: "F023", actual_type: "PRESCRIPTION", content: {
          doctor_name: "Dr. P. Banerjee", doctor_registration: "WB/34567/2015",
          diagnosis: "Morbid Obesity — BMI 37",
          treatment: "Bariatric Consultation and Customised Diet Plan" } },
        { file_id: "F024", actual_type: "HOSPITAL_BILL", content: {
          line_items: [
            { description: "Bariatric Consultation", amount: 3000 },
            { description: "Personalised Diet and Nutrition Program", amount: 5000 } ],
          total: 8000 } },
      ],
    },
  },
];
