"""Fixed P0 offline clauses. Evidence cannot choose or weaken this policy."""

from dataclasses import asdict, dataclass
import json


POLICY_VERSION = "p0-offline-1"
T006_NODE = "tests/unit/test_guide_quality.py::GuideQualityTests::test_rgb_psnr_and_ssim_on_all_frames_outside_edge_band"
METRIC_CROPS = ((574, 720, 0, 0), (574, 720, 706, 0), (132, 294, 574, 0),
                (132, 294, 574, 426), (124, 124, 578, 298))


@dataclass(frozen=True)
class ClauseSpec:
    clause_id: str
    kind: str
    required_nodes: tuple[str, ...]
    missing_code: str | None = None


P0_CASES = (
    ('T001', (
        ClauseSpec('T001.1', 'component', (
            'tests/unit/test_media.py::MediaTests::test_real_output_has_exact_media_format_and_timebase',
            'tests/unit/test_media.py::MediaTests::test_original_bytes_and_hashes_are_preserved',
            'tests/unit/test_media.py::MediaTests::test_only_the_first_five_seconds_are_used',
            'tests/unit/test_media.py::MediaTests::test_portrait_is_pillarboxed_without_crop',
        )),
    )),
    ('T002', (
        ClauseSpec('T002.1', 'component', (
            'tests/unit/test_asset.py::AssetTests::test_original_hash_and_partial_alpha_are_preserved',
            'tests/unit/test_asset.py::AssetTests::test_binary_alpha_is_valid',
            'tests/unit/test_asset.py::AssetTests::test_opaque_empty_and_nonzero_minimum_alpha_are_rejected',
            'tests/unit/test_asset.py::AssetTests::test_missing_and_corrupt_files_have_safe_errors',
            'tests/unit/test_asset.py::AssetTests::test_oversized_file_is_rejected_before_decode',
            'tests/unit/test_asset.py::AssetTests::test_oversized_dimensions_are_rejected_before_decode',
        )),
    )),
    ('T003', (
        ClauseSpec('T003.1', 'component', (
            'tests/unit/test_placement.py::PlacementTests::test_center_and_aspect_ratio',
            'tests/unit/test_placement.py::PlacementTests::test_invalid_fractions_are_rejected',
            'tests/unit/test_placement.py::PlacementTests::test_zero_height_and_rotation_are_rejected',
            'tests/unit/test_keyframe.py::KeyframeTests::test_exact_alpha_blend_matches_independent_reference',
            'tests/unit/test_keyframe.py::KeyframeTests::test_repeated_render_has_identical_png_hash',
        )),
    )),
    ('T004', (
        ClauseSpec('T004.1', 'component', (
            'tests/unit/test_keyframe.py::KeyframeTests::test_exact_alpha_blend_matches_independent_reference',
            'tests/unit/test_guidance.py::GuidanceTests::test_real_compilation_preserves_inputs_and_renders_every_frame',
        )),
    )),
    ('T005', (
        ClauseSpec('T005.1', 'component', (
            'tests/unit/test_guidance.py::GuidanceTests::test_real_compilation_preserves_inputs_and_renders_every_frame',
        )),
    )),
    ('T006', (
        ClauseSpec('T006.1', 'component', (
            'tests/unit/test_guide_quality.py::GuideQualityTests::test_rgb_psnr_and_ssim_on_all_frames_outside_edge_band',
        )),
    )),
    ('T007', (
        ClauseSpec('T007.1', 'component', (
            'tests/unit/test_runway.py::RunwayTests::test_modes_send_exact_payloads_through_the_pinned_sdk',
            'tests/unit/test_frozen_request.py::FrozenRequestTests::test_mode_b_uses_its_frozen_guide_and_omits_keyframes',
            'tests/unit/test_frozen_request.py::FrozenRequestTests::test_each_frozen_input_changes_the_reservation_fingerprint',
        )),
    )),
    ('T008', (
        ClauseSpec('T008.1', 'component', (
            'tests/unit/test_runway.py::RunwayTests::test_retryable_http_errors_make_one_post_each',
            'tests/unit/test_runway.py::RunwayTests::test_connection_failures_and_timeouts_make_one_post_each',
        )),
    )),
    ('T009', (
        ClauseSpec('T009.1', 'component', (
            'tests/unit/test_budget.py::BudgetTests::test_twenty_concurrent_requests_reserve_only_one_available_charge',
        )),
    )),
    ('T010', (
        ClauseSpec('T010.1', 'component', (
            'tests/unit/test_budget.py::BudgetTests::test_reservation_uses_approved_quote_and_persists_across_restart',
            'tests/unit/test_budget.py::BudgetTests::test_same_key_with_changed_payload_conflicts',
            'tests/unit/test_submit_command.py::SubmitCommandTests::test_restores_exact_request_and_repeated_command_needs_no_key',
        )),
        ClauseSpec('T010.2', 'api', (), 'IMPLEMENTATION_MISSING'),
    )),
    ('T011', (
        ClauseSpec('T011.1', 'component', (
            'tests/unit/test_submission.py::SubmissionTests::test_submitting_is_committed_before_the_create_call',
            'tests/unit/test_submission.py::SubmissionTests::test_process_exit_recovers_unknown_without_replay',
        )),
    )),
    ('T012', (
        ClauseSpec('T012.1', 'component', (
            'tests/unit/test_retrieve.py::RetrievalTests::test_sdk_get_timeout_resumes_after_restart_without_another_create',
        )),
    )),
    ('T013', (
        ClauseSpec('T013.1', 'component', (
            'tests/unit/test_submission.py::SubmissionTests::test_timeout_keeps_money_and_prevents_replay',
            'tests/unit/test_charges.py::ChargeTests::test_final_charge_replaces_reservation_instead_of_adding_to_it',
            'tests/unit/test_charges.py::ChargeTests::test_concurrent_reconciliation_cannot_double_spend_released_money',
            'tests/unit/test_run_audit.py::RunAuditTests::test_refunds_never_restore_the_attempt_count',
        )),
    )),
    ('T014', (
        ClauseSpec('T014.1', 'component', (
            'tests/unit/test_prepare.py::PrepareTests::test_second_placement_keeps_first_attempt_unchanged',
            'tests/unit/test_review_evidence.py::ReviewEvidenceTests::test_repeat_and_other_attempt_keep_existing_review_bytes_unchanged',
            'tests/unit/test_review_page.py::ReviewPageTests::test_page_is_immutable_repeatable_and_independent_of_another_attempt',
        )),
        ClauseSpec('T014.2', 'workflow', (), 'IMPLEMENTATION_MISSING'),
    )),
    ('T015', (
        ClauseSpec('T015.1', 'component', (
            'tests/unit/test_output_completion.py::OutputCompletionTests::test_artifacts_precede_success_and_fixture_content_never_claims_workflow_pass',
            'tests/unit/test_output_completion.py::OutputCompletionTests::test_corrupt_download_cannot_create_artifacts_or_local_success',
            'tests/unit/test_raw_video.py::RawVideoTests::test_corrupt_late_audio_is_rejected_even_when_video_and_metadata_pass',
        )),
        ClauseSpec('T015.2', 'component', (
            'tests/unit/test_runtime_evidence.py::RuntimeEvidenceTests::test_injected_constructed_sdk_client_cannot_claim_default_transport',
        )),
        ClauseSpec('T015.3', 'qa', (), 'IMPLEMENTATION_MISSING'),
        ClauseSpec('T015.4', 'runtime_evidence', (), 'EXTERNAL_EVIDENCE_REQUIRED'),
        ClauseSpec('T015.5', 'component', (
            'tests/unit/test_runtime_chain.py::RuntimeChainTests::test_injected_receipts_keep_valid_bindings_but_fail_default_network',
            'tests/unit/test_runtime_chain.py::RuntimeChainTests::test_relabelled_mock_receipts_cannot_replace_retained_pins',
        )),
    )),
    ('T016', (
        ClauseSpec('T016.1', 'component', (
            'tests/unit/test_review_evidence.py::ReviewEvidenceTests::test_mode_a_reuses_actual_static_keyframe_and_preserves_original_source_reference',
            'tests/unit/test_review_evidence.py::ReviewEvidenceTests::test_mode_b_samples_real_24_fps_guide_and_30_fps_raw_frames',
            'tests/unit/test_alignment.py::AlignmentTests::test_24_fps_source_matches_30_fps_raw_without_changing_timestamps',
            'tests/unit/test_raw_origin.py::RawOriginTests::test_alignment_uses_elapsed_nearest_ties_span_and_source_epsilon',
        )),
        ClauseSpec('T016.2', 'viewer', (
            'tests/ui/test_review_page_browser.py::ReviewPageBrowserTests::test_native_views_samples_and_playback_stay_within_one_local_attempt',
        )),
    )),
    ('T017', (
        ClauseSpec('T017.1', 'qa', (), 'IMPLEMENTATION_MISSING'),
    )),
    ('T018', (
        ClauseSpec('T018.1', 'component', (
            'tests/unit/test_scorecard.py::ScorecardTests::test_complete_declared_review_can_pass_with_two_or_more_plays',
            'tests/unit/test_scorecard.py::ScorecardTests::test_each_dimension_must_pass_without_averaging',
            'tests/unit/test_scorecard.py::ScorecardTests::test_absent_or_unsigned_valid_draft_is_not_run',
            'tests/unit/test_scorecard.py::ScorecardTests::test_scores_plays_inspected_times_and_required_observation_are_strict',
            'tests/unit/test_scorecard.py::ScorecardTests::test_nonblank_review_time_must_be_an_actual_utc_timestamp',
        )),
        ClauseSpec('T018.2', 'human_review', (), 'EXTERNAL_EVIDENCE_REQUIRED'),
    )),
    ('T019', (
        ClauseSpec('T019.1', 'component', (
            'tests/unit/test_prepare.py::PrepareTests::test_real_cli_exports_verified_snapshot_without_secrets_or_paid_claims',
            'tests/unit/test_export_inventory.py::ExportInventoryTests::test_no_directory_scan_or_private_url_secret_html_or_other_attempt_is_selected',
            'tests/unit/test_export_archive.py::ExportArchiveTests::test_real_modes_and_selections_preserve_exact_members_crc_and_fixed_metadata',
        )),
        ClauseSpec('T019.2', 'qa', (), 'IMPLEMENTATION_MISSING'),
    )),
    ('T020', (
        ClauseSpec('T020.1', 'component', (
            'tests/unit/test_run_start.py::RunDeadlineTests::test_exact_experiment_deadline_blocks_final_send_and_stays_stopped',
            'tests/unit/test_budget.py::BudgetTests::test_money_and_attempt_caps_are_independent',
            'tests/unit/test_run_audit.py::RunAuditTests::test_literal_p0_and_reduced_caps_apply_at_each_spending_boundary',
            'tests/unit/test_charges.py::ChargeTests::test_over_budget_billing_is_retained_and_stops_new_reservations',
            'tests/unit/test_checked_send.py::CheckedSendTests::test_charge_over_quote_before_send_blocks_new_spending',
        )),
        ClauseSpec('T020.2', 'api', (), 'IMPLEMENTATION_MISSING'),
    )),
    ('T021', (
        ClauseSpec('T021.1', 'component', (
            'tests/unit/test_media.py::MediaTests::test_bad_paths_and_corrupt_media_have_safe_errors',
            'tests/unit/test_asset.py::AssetTests::test_opaque_empty_and_nonzero_minimum_alpha_are_rejected',
            'tests/unit/test_placement.py::PlacementTests::test_invalid_fractions_are_rejected',
            'tests/unit/test_account.py::AccountTests::test_balance_and_model_limits_block_the_account_check',
            'tests/unit/test_runway.py::RunwayTests::test_retryable_http_errors_make_one_post_each',
        )),
        ClauseSpec('T021.2', 'api', (), 'IMPLEMENTATION_MISSING'),
    )),
    ('T022', (
        ClauseSpec('T022.1', 'component', (
            'tests/unit/test_archive_attempt_facts.py::ArchiveAttemptFactsTests::test_invalid_references_missing_and_corrupt_payloads_are_rejected_safely',
            'tests/unit/test_archive_attempt_facts.py::ArchiveAttemptFactsTests::test_selected_unsigned_cards_and_invalid_bindings_are_rejected',
            'tests/unit/test_archive_attempt_facts.py::ArchiveAttemptFactsTests::test_task_raw_and_private_references_must_match_the_inventory',
            'tests/unit/test_run_audit.py::RunAuditTests::test_duplicate_financial_and_journal_records_fail_with_localized_immutable_issues',
            'tests/unit/test_run_audit.py::RunAuditTests::test_known_invalid_money_still_fails_when_approval_or_tables_are_missing',
        )),
        ClauseSpec('T022.2', 'qa', (), 'IMPLEMENTATION_MISSING'),
    )),
    ('T023', (
        ClauseSpec('T023.1', 'viewer', (
            'tests/ui/test_review_page_browser.py::ReviewPageBrowserTests::test_native_views_samples_and_playback_stay_within_one_local_attempt',
        )),
        ClauseSpec('T023.2', 'workflow', (), 'IMPLEMENTATION_MISSING'),
        ClauseSpec('T023.3', 'runtime_evidence', (), 'EXTERNAL_EVIDENCE_REQUIRED'),
    )),
)


def policy_bytes() -> bytes:
    """Bind exact node IDs, explicit gaps, and the fixed codec metric rules."""
    data = {"version": POLICY_VERSION,
            "cases": [{"id": case, "clauses": [asdict(clause) for clause in clauses]}
                      for case, clauses in P0_CASES],
            "codec": {"node": T006_NODE, "frames": 120, "color": "8-bit RGB",
                      "edgeExclusionPixels": 2, "pixelsPerFrame": 919552,
                      "crops": METRIC_CROPS, "minimumPsnrDb": 30, "minimumSsim": 0.95}}
    return (json.dumps(data, sort_keys=True, indent=2) + "\n").encode("utf-8")
