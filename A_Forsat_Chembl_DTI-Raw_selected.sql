-- اجرا در سرور شما: این خروجی را در دیتابیس ForsatKG می‌سازد
USE [ForsatKG];
GO

IF OBJECT_ID('dbo.Forsat_Chembl_DTI_Raw', 'U') IS NOT NULL
    DROP TABLE dbo.Forsat_Chembl_DTI_Raw;
GO

;WITH base AS (
    SELECT
        act.activity_id,
        act.assay_id,
        act.doc_id,
        act.record_id,
        act.molregno,
        md.chembl_id AS drug_chembl_id,
        md.pref_name AS drug_name,
        cs.canonical_smiles,
        cp.full_mwt,
        a.tid AS target_tid,
        td.chembl_id AS target_chembl_id,
        td.pref_name AS target_name,
        tc.component_id,
        prot.accession AS uniprot_id,
        -- raw activity fields (prefer standard_*, fallback to raw)
        act.standard_type,
        act.standard_value,
        act.standard_units,
        act.standard_relation,
        act.value,
        act.units,
        act.relation,
        act.text_value,
        act.standard_text_value,
        act.pchembl_value,
        act.standard_flag,
        act.action_type,
        act.activity_comment,
        act.data_validity_comment,
        act.src_id AS activity_src_id,
        a.description AS assay_description,
        a.assay_type,
        a.assay_category,
        a.assay_test_type,
        a.src_id AS assay_src_id,
        d.year AS doc_year,
        d.pubmed_id,
        s.src_short_name AS source_short_name,
        -- drug_mechanism action_type (if present)
        dm.action_type AS mech_action_type,
        -- aggregated activity_properties (type=val;type2=val2;...)
        STUFF(
            (
                SELECT ';' + ISNULL(ap2.[type],'') + '=' +
                       COALESCE(CASE WHEN ap2.value IS NOT NULL THEN CAST(ap2.value AS varchar(60))
                                     WHEN ap2.text_value IS NOT NULL THEN ap2.text_value
                                     ELSE ap2.standard_text_value END, '')
                FROM [ChEMBL].dbo.activity_properties ap2
                WHERE ap2.activity_id = act.activity_id
                FOR XML PATH(''), TYPE
            ).value('.', 'NVARCHAR(MAX)'), 1, 1, ''
        ) AS activity_properties_raw
    FROM [ChEMBL].dbo.activities act
    INNER JOIN [ChEMBL].dbo.molecule_dictionary md ON act.molregno = md.molregno
    LEFT JOIN [ChEMBL].dbo.compound_structures cs    ON md.molregno  = cs.molregno
    LEFT JOIN [ChEMBL].dbo.compound_properties cp   ON md.molregno  = cp.molregno
    LEFT JOIN [ChEMBL].dbo.assays a                  ON act.assay_id = a.assay_id
    LEFT JOIN [ChEMBL].dbo.target_dictionary td      ON a.tid = td.tid
    LEFT JOIN [ChEMBL].dbo.target_components tc      ON td.tid = tc.tid
    LEFT JOIN [ChEMBL].dbo.component_sequences prot  ON tc.component_id = prot.component_id
    LEFT JOIN [ChEMBL].dbo.docs d                    ON act.doc_id = d.doc_id
    LEFT JOIN [ChEMBL].dbo.source s                  ON act.src_id = s.src_id
    -- add drug_mechanism (may have action_type info)
    LEFT JOIN [ChEMBL].dbo.drug_mechanism dm         ON md.molregno = dm.molregno AND td.tid = dm.tid
    WHERE
        md.chembl_id IS NOT NULL
        -- شما می‌توانید شرط‌های فیلتر اضافه کنید (مثل md.max_phase = 4) در اینجا
), unit_conv AS (
    SELECT
        b.*,
        -- pick a canonical numeric_value: prefer standard_value then fallback to value
        COALESCE(b.standard_value, b.value) AS numeric_value,
        COALESCE(NULLIF(LTRIM(RTRIM(b.standard_units)),''), NULLIF(LTRIM(RTRIM(b.units)),'')) AS raw_units,
        COALESCE(NULLIF(LTRIM(RTRIM(b.standard_relation)),''), NULLIF(LTRIM(RTRIM(b.relation)),'')) AS raw_relation
    FROM base b
), molar_calc AS (
    SELECT
        uc.*,

        -- try convert when standard units already in molar units (pM/nM/µM/mM/M)
        CASE
            WHEN uc.numeric_value IS NULL THEN NULL

            WHEN LOWER(uc.raw_units) LIKE '%pm%' OR LOWER(uc.raw_units) LIKE '%picomolar%' THEN CAST(uc.numeric_value * 1e-12 AS FLOAT)
            WHEN LOWER(uc.raw_units) LIKE '%nm%' OR LOWER(uc.raw_units) LIKE '%nanomolar%' THEN CAST(uc.numeric_value * 1e-9 AS FLOAT)
            WHEN LOWER(uc.raw_units) LIKE '%µm%' OR LOWER(uc.raw_units) LIKE '%um%' OR LOWER(uc.raw_units) LIKE '%micromolar%' OR LOWER(uc.raw_units) LIKE '%μm%' THEN CAST(uc.numeric_value * 1e-6 AS FLOAT)
            WHEN LOWER(uc.raw_units) LIKE '%mm%' OR LOWER(uc.raw_units) LIKE '%millimolar%' THEN CAST(uc.numeric_value * 1e-3 AS FLOAT)
            WHEN LOWER(uc.raw_units) LIKE '%10^-12m%' THEN CAST(uc.numeric_value * 1e-12 AS FLOAT)
            WHEN LOWER(uc.raw_units) LIKE '%10^-9m%' THEN CAST(uc.numeric_value * 1e-9 AS FLOAT)
            WHEN LOWER(uc.raw_units) LIKE '%10^-6m%' THEN CAST(uc.numeric_value * 1e-6 AS FLOAT)
            WHEN LOWER(uc.raw_units) LIKE '%mol/l%' OR LOWER(uc.raw_units) LIKE '%mol l-1%' THEN CAST(uc.numeric_value AS FLOAT)

            -- mass concentration -> convert to molar using full_mwt when available
            WHEN LOWER(uc.raw_units) LIKE '%ng/ml%' THEN
                 CASE WHEN uc.full_mwt IS NOT NULL AND uc.full_mwt > 0
                      THEN CAST(uc.numeric_value * 1e-6 / uc.full_mwt AS FLOAT) ELSE NULL END
            WHEN LOWER(uc.raw_units) LIKE '%ug/ml%' OR LOWER(uc.raw_units) LIKE '%µg/ml%' THEN
                 CASE WHEN uc.full_mwt IS NOT NULL AND uc.full_mwt > 0
                      THEN CAST(uc.numeric_value * 1e-3 / uc.full_mwt AS FLOAT) ELSE NULL END
            WHEN LOWER(uc.raw_units) LIKE '%mg/ml%' THEN
                 CASE WHEN uc.full_mwt IS NOT NULL AND uc.full_mwt > 0
                      THEN CAST(uc.numeric_value * 1.0 / uc.full_mwt AS FLOAT) ELSE NULL END

            ELSE NULL
        END AS molar_value,

        -- flag whether we could convert to molar
        CASE
            WHEN uc.numeric_value IS NULL THEN 0
            WHEN
                LOWER(uc.raw_units) LIKE '%pm%' OR LOWER(uc.raw_units) LIKE '%nm%' OR LOWER(uc.raw_units) LIKE '%µm%' OR
                LOWER(uc.raw_units) LIKE '%um%' OR LOWER(uc.raw_units) LIKE '%mm%' OR LOWER(uc.raw_units) LIKE '%mol/l%' OR
                LOWER(uc.raw_units) LIKE '%ng/ml%' OR LOWER(uc.raw_units) LIKE '%ug/ml%' OR LOWER(uc.raw_units) LIKE '%mg/ml%' OR
                LOWER(uc.raw_units) LIKE '%10^-12m%' OR LOWER(uc.raw_units) LIKE '%10^-9m%' OR LOWER(uc.raw_units) LIKE '%10^-6m%'
            THEN 1
            ELSE 0
        END AS could_convert_molar,

        -- censor information
        CASE WHEN uc.raw_relation IS NOT NULL AND (uc.raw_relation LIKE '%<%' OR uc.raw_relation LIKE '%>%' ) THEN 1
             WHEN uc.standard_relation IS NOT NULL AND (uc.standard_relation LIKE '%<%' OR uc.standard_relation LIKE '%>%') THEN 1
             ELSE 0 END AS is_censored,

        -- pick a censor sign text (if any)
        CASE
            WHEN uc.standard_relation IS NOT NULL AND (uc.standard_relation LIKE '%<%' OR uc.standard_relation LIKE '%>%') THEN uc.standard_relation
            WHEN uc.relation IS NOT NULL AND (uc.relation LIKE '%<%' OR uc.relation LIKE '%>%') THEN uc.relation
            ELSE NULL
        END AS censor_sign,

        -- suggested assay confidence mapping (editable later)
        CASE
            WHEN UPPER(uc.standard_type) IN ('KD','KI') THEN 1.00
            WHEN UPPER(uc.standard_type) IN ('IC50') THEN 0.80
            WHEN UPPER(uc.standard_type) IN ('EC50','AC50') THEN 0.70
            WHEN UPPER(uc.standard_type) IN ('POTENCY') THEN 0.50
            ELSE 0.60
        END AS assay_confidence_suggested

    FROM unit_conv uc
), px_calc AS (
    SELECT
        mc.*,
        -- compute pX when molar_value available and >0
        CASE WHEN mc.molar_value IS NOT NULL AND mc.molar_value > 0
             THEN CAST(-LOG10(mc.molar_value) AS FLOAT)
             ELSE NULL END AS pX_value
    FROM molar_calc mc
)

SELECT
    -- basic identifiers
    activity_id, assay_id, doc_id, record_id, molregno,
    drug_chembl_id, drug_name, canonical_smiles, full_mwt,
    target_tid, target_chembl_id, target_name, component_id, uniprot_id,

    -- raw activity fields
    standard_type, standard_value, standard_units, standard_relation,
    value, units, relation, text_value, standard_text_value, pchembl_value,
    standard_flag, action_type, mech_action_type, activity_comment, data_validity_comment,
    activity_src_id, assay_description, assay_type, assay_category, assay_test_type,
    assay_src_id, doc_year, pubmed_id, source_short_name,

    -- aggregated properties
    activity_properties_raw,

    -- conversion & flags
    numeric_value, raw_units, raw_relation,
    molar_value,
    could_convert_molar,
    is_censored,
    censor_sign,
    pX_value,
    assay_confidence_suggested,

    -- convenience: quality flag: 1 = good (molar available), 0 = need manual handling
    CASE WHEN could_convert_molar = 1 THEN 1 ELSE 0 END AS conversion_quality_flag,

    -- NEW: inferred / explicit interaction type (Activation / Inhibition / Unknown)
    CASE
        -- prefere explicit action_type from activities
        WHEN action_type IS NOT NULL AND (
             UPPER(action_type) LIKE '%AGONIST%' OR UPPER(action_type) LIKE '%ACTIVA%' OR UPPER(action_type) LIKE '%AGON%' )
             THEN 'Activation'
        WHEN action_type IS NOT NULL AND (
             UPPER(action_type) LIKE '%ANTAGONIST%' OR UPPER(action_type) LIKE '%INHIBIT%' OR UPPER(action_type) LIKE '%BLOCKER%' )
             THEN 'Inhibition'

        -- then drug_mechanism.action_type if present
        WHEN mech_action_type IS NOT NULL AND (
             UPPER(mech_action_type) LIKE '%AGONIST%' OR UPPER(mech_action_type) LIKE '%ACTIVA%' OR UPPER(mech_action_type) LIKE '%AGON%' )
             THEN 'Activation'
        WHEN mech_action_type IS NOT NULL AND (
             UPPER(mech_action_type) LIKE '%ANTAGONIST%' OR UPPER(mech_action_type) LIKE '%INHIBIT%' OR UPPER(mech_action_type) LIKE '%BLOCKER%' )
             THEN 'Inhibition'

        -- infer from assay readout type
        WHEN UPPER(standard_type) IN ('IC50','KI','KD') THEN 'Inhibition'
        WHEN UPPER(standard_type) IN ('EC50','AC50','POTENCY') THEN 'Activation'

        -- infer from aggregated properties text
        WHEN LOWER(activity_properties_raw) LIKE '%agonist%' OR LOWER(activity_properties_raw) LIKE '%activator%' THEN 'Activation'
        WHEN LOWER(activity_properties_raw) LIKE '%antagonist%' OR LOWER(activity_properties_raw) LIKE '%inhibitor%' THEN 'Inhibition'

        ELSE 'Unknown'
    END AS interaction_type,

    -- NEW: source of the interaction_type decision
    CASE
        WHEN action_type IS NOT NULL AND (
             UPPER(action_type) LIKE '%AGONIST%' OR UPPER(action_type) LIKE '%ACTIVA%' OR UPPER(action_type) LIKE '%AGON%' OR
             UPPER(action_type) LIKE '%ANTAGONIST%' OR UPPER(action_type) LIKE '%INHIBIT%' OR UPPER(action_type) LIKE '%BLOCKER%')
             THEN 'activities.action_type'
        WHEN mech_action_type IS NOT NULL AND (
             UPPER(mech_action_type) LIKE '%AGONIST%' OR UPPER(mech_action_type) LIKE '%ACTIVA%' OR UPPER(mech_action_type) LIKE '%AGON%' OR
             UPPER(mech_action_type) LIKE '%ANTAGONIST%' OR UPPER(mech_action_type) LIKE '%INHIBIT%' OR UPPER(mech_action_type) LIKE '%BLOCKER%')
             THEN 'drug_mechanism.action_type'
        WHEN UPPER(standard_type) IN ('IC50','KI','KD') THEN 'inferred_from_standard_type'
        WHEN UPPER(standard_type) IN ('EC50','AC50','POTENCY') THEN 'inferred_from_standard_type'
        WHEN LOWER(activity_properties_raw) LIKE '%agonist%' OR LOWER(activity_properties_raw) LIKE '%activator%' OR
             LOWER(activity_properties_raw) LIKE '%antagonist%' OR LOWER(activity_properties_raw) LIKE '%inhibitor%' THEN 'inferred_from_activity_properties'
        ELSE 'unknown_source'
    END AS interaction_source

INTO dbo.Forsat_Chembl_DTI_Raw
FROM px_calc
WHERE
    -- حداقل یک مقدار عددی موجود باشد
    (numeric_value IS NOT NULL)
    -- و انواع readout مورد نظر را فیلتر می‌کنیم (به دلخواه شما)
    --AND UPPER(standard_type) IN ('IC50','KI','KD','EC50','AC50','POTENCY')
;
GO
drop table if exists [Forsat_Chembl_DTI_Raw_selected];
select * into [Forsat_Chembl_DTI_Raw_selected] from(
SELECT
      [drug_chembl_id]
      ,[canonical_smiles]
      ,[target_chembl_id]
      ,[target_name]
      ,[component_id]
      ,[uniprot_id]
      ,[standard_type]
      ,[standard_value]
      ,[standard_flag]
      ,[assay_description]
      ,[assay_type]
      ,[assay_category]
      ,[pX_value]
      ,[assay_confidence_suggested]
      ,[conversion_quality_flag]
      ,[interaction_type]
  FROM [ForsatKG].[dbo].[Forsat_Chembl_DTI_Raw]
  where standard_flag = 1
  and   Uniprot_id is not null
  and drug_chembl_id is not null
  and assay_confidence_suggested > 0.5
  and standard_type in ('IC50','KI','KD','EC50')
  and pX_value is not null
  )a




  select * into Forsat_Chembl_DTI_Raw_Drug_embedding from(
  select distinct drug_chembl_id, canonical_smiles from Forsat_Chembl_DTI_Raw_selected where canonical_smiles is not null)a


  select * into Forsat_Chembl_DTI_Raw_Protein_embedding from(
    SELECT distinct      [uniprot_id], [protein_sequence]
	FROM [ForsatKG].[dbo].[ChEMBL_DTI] where protein_sequence is not null)a

