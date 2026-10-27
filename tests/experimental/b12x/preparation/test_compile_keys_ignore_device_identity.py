"""Production compiler factories share artifacts across matching CUDA devices."""

import multiprocessing
import traceback


def _probe(ordinal, uuid, capability, sm_count, connection):
    try:
        import torch
        from b12x._lib.compile_pool import CompileJob, _initialize_worker, _factory
        from b12x._lib.compile_plan import plan_compilations
        from b12x.preparation import FrozenMapping

        context = multiprocessing.get_context("spawn")
        _initialize_worker(
            ordinal,
            capability,
            uuid,
            "NVIDIA test target",
            sm_count,
            101376,
            102400,
            context.Array("q", (0, 0)),
        )
        mamba = dict(
            heads=4,
            groups=1,
            head_dim=64,
            dstate=64,
            max_tokens=16,
            max_seqs=2,
            max_slots=4,
            x_stride=256,
            b_stride=64,
            c_stride=64,
            dt_stride=4,
            state_stride=16384,
            state_dtype="float32",
            dt_dtype="bfloat16",
            d_dtype="float32",
            bias_dtype="float32",
            index_dtype="int32",
        )
        mamba = FrozenMapping(mamba)
        jobs = {
            "vision": CompileJob.create(
                "b12x.norm.vision:_compile", "gelu", 128, 1, 2, ordinal
            ),
            "split_k": CompileJob.create(
                "b12x.gemm.blockscaled._reduce:compile_reduce", 128, 2, ordinal
            ),
            "block_fp8": CompileJob.create(
                "b12x.gemm.blockscaled._block_fp8_gemv:compile_block_fp8_gemv",
                ordinal,
                128,
                256,
            ),
            "iq2_xs": CompileJob.create(
                "b12x.gemm.blockscaled._iq2_xs_gemv:compile_gemv", ordinal, 128, 256, 4
            ),
            "selection": CompileJob.create(
                "b12x.attention.qsa._stable_select_cute:compile_stable_selection",
                64,
                ordinal,
            ),
            "rotation": CompileJob.create(
                "b12x.attention.compressed_sparse_mla.rotary:compile_rotation",
                dict(max_rows=16, heads=8, dim=512),
                ordinal,
            ),
            "cast": CompileJob.create(
                "b12x.attention.compressed_sparse_mla.cast:_compile_cast",
                torch.bfloat16,
                torch.float32,
                ordinal,
            ),
            "index_weights": CompileJob.create(
                "b12x.attention.compressed_sparse_mla.weight_scale:compile_scale",
                ordinal,
            ),
            "compress": CompileJob.create(
                "b12x.attention.mla_compress._cute:compile_compress",
                1,
                16,
                4,
                4,
                ordinal,
            ),
            "indexer": CompileJob.create(
                "b12x.attention.dsa_indexer.mxfp4:_compile",
                "sort",
                (64, False),
                ordinal,
            ),
            "embedding": CompileJob.create(
                "b12x.sequence.embedding._kernel:compile_embedding",
                128,
                torch.bfloat16,
                torch.int32,
                ordinal,
            ),
            "mamba_scan": CompileJob.create(
                "b12x.sequence.mamba_ssd._kernel:compile_scan", mamba, ordinal
            ),
            "mamba_chunked": CompileJob.create(
                "b12x.sequence.mamba_ssd._chunked:compile_chunked",
                mamba,
                FrozenMapping(dict(prepare_warps=4, carry_elements=1, output_rows=16)),
                ordinal,
            ),
            "w4a8": CompileJob.create(
                "b12x.moe._shared.kernels.w4a8_compact_micro:_compiled_direct_w4a8_compact",
                ordinal,
                4,
                4,
                512,
                512,
                8,
                torch.int32,
                1,
                1,
                None,
                False,
            ),
            "w4a16_micro": CompileJob.create(
                "b12x.moe._shared.kernels.w4a16.kernel.direct_launch:_compile_w4a16_small_m_direct",
                m=4,
                hidden_size=512,
                intermediate_size=512,
                num_experts=8,
                topk=4,
                activation="silu",
                fast_math=False,
                topk_ids_dtype=torch.int32,
                device=torch.device("cuda", ordinal),
            ),
            "w4a16_fc2": CompileJob.create(
                "b12x.moe._shared.kernels.w4a16.kernel.direct_launch:_compile_w4a16_fc2_direct",
                hidden_size=512,
                intermediate_size=512,
                num_experts=16,
                topk_ids_dtype=torch.int32,
                device=torch.device("cuda", ordinal),
            ),
        }
        from b12x.gemm._shared.mxfp8_bmm import _BMajor

        jobs["mxfp8_bmm"] = CompileJob.create(
            "b12x.gemm._shared.mxfp8_bmm:_compile",
            b_major=_BMajor.N,
            groups=8,
            m=16,
            n=256,
            k=512,
            device=torch.device("cuda", ordinal),
        )
        jobs["mxfp8_mla"] = CompileJob.create(
            "b12x.gemm._shared.mxfp8_bmm:_compile_mla_query_projection",
            output_fp8=False,
            b_major=_BMajor.N,
            groups=8,
            m=16,
            n=256,
            k=512,
            device=torch.device("cuda", ordinal),
        )
        jobs["w4a16_gemm"] = CompileJob.create(
            "b12x.moe._shared.kernels.w4a16.kernel.gemm_launch:compile_w4a16_gemm",
            size_m=4,
            size_n=512,
            size_k=512,
            num_experts=8,
            top_k=4,
            mul_topk_weights=False,
            tile_n=128,
            tile_k=128,
            moe_block_size=8,
            max_m_blocks=8,
        )
        jobs["w4a16_fused"] = CompileJob.create(
            "b12x.moe._shared.kernels.w4a16.kernel.fused_launch:compile_w4a16_fused_moe",
            size_m=4,
            hidden_size=512,
            intermediate_size=512,
            num_experts=8,
            top_k=4,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            max_m_blocks=8,
            sms=sm_count,
            max_shared_mem=101376,
        )
        from dataclasses import replace
        from b12x.norm.hyperconnection._tuning import HyperConnectionQuery, TUNING

        for operation in (
            "grouped_rmsnorm",
            "scaled_silu",
            "combine",
            "combine_norm",
            "engram_mix",
            "swiglu",
            "relu2",
            "add",
            "sigmoid",
            "head_gate",
        ):
            query = HyperConnectionQuery(
                dtype="bfloat16",
                max_tokens=128,
                hidden_size=2560,
                streams=4,
                lowrank=320,
                operation=operation,
                output_mode="provided",
                zero_centered=False,
                weight_dtype="float32",
            )
            variants = (
                ((2560, 8), (2560, 10), (2560, 16), (2559, 8))
                if operation == "combine_norm"
                else ((2560, None),)
            )
            for hidden, warps in variants:
                variant = replace(query, hidden_size=hidden)
                config = TUNING.default_config(variant, None)
                if warps is not None:
                    config = replace(config, reduction_num_warps=warps)
                payload = TUNING.encode_query(variant)
                payload.pop("autotune_effort")
                jobs[f"hyperconnection_{operation}_{hidden}_{warps}"] = (
                    CompileJob.create(
                        "b12x.norm.hyperconnection._preparation:_compile_hyperconnection",
                        payload,
                        config.to_dict(),
                        ordinal,
                    )
                )

        results = {}
        for name, job in jobs.items():
            try:
                with plan_compilations() as programs:
                    _factory(job.factory)(*job.args, **dict(job.kwargs))
                assert programs, name
                results[name] = sorted(
                    (program.dialect, program.key) for program in programs
                )
            except Exception as error:
                raise RuntimeError(
                    f"compiler factory {name} failed: {error}"
                ) from error
        from b12x.norm.hyperconnection import _cute

        with plan_compilations() as programs:
            _cute._compile(
                ("norm", 4, 128),
                _cute._GroupedRmsNorm(4, 128, True),
                3,
                torch.device("cuda", ordinal),
                has_eps=True,
                runtime_ints=1,
                runtime_int64s=1,
            )
        assert programs
        results["hyperconnection"] = sorted((p.dialect, p.key) for p in programs)
        assert not torch.cuda.is_initialized()
        connection.send((results, None))
    except BaseException:
        connection.send((None, traceback.format_exc()))
    finally:
        connection.close()


def test_program_keys_ignore_ordinals_and_uuids_but_preserve_hardware_geometry():
    context = multiprocessing.get_context("spawn")
    targets = (
        (0, "GPU-first", (12, 0), 188),
        (7, "GPU-second", (12, 0), 188),
        (3, "GPU-third", (12, 1), 188),
        (4, "GPU-fourth", (12, 0), 170),
    )
    processes, readers = [], []
    try:
        for target in targets:
            reader, writer = context.Pipe(duplex=False)
            process = context.Process(target=_probe, args=(*target, writer))
            process.start()
            writer.close()
            processes.append(process)
            readers.append(reader)
        results = []
        for process, reader in zip(processes, readers, strict=True):
            assert reader.poll(120), f"compiler probe {process.pid} timed out"
            result, error = reader.recv()
            assert error is None, error
            results.append(result)
            process.join(timeout=10)
            assert process.exitcode == 0
        assert results[0] == results[1], [
            name for name in results[0] if results[0][name] != results[1][name]
        ]
        assert all(results[0][name] != results[2][name] for name in results[0])
        for name, programs in results[0].items():
            # Triton targets encode compute capability; CuTe also keys SM count.
            if all(dialect == "triton" for dialect, _ in programs):
                assert programs == results[3][name], name
            else:
                assert programs != results[3][name], name
        assert (
            len(
                {
                    tuple(results[0][f"hyperconnection_combine_norm_2560_{warps}"])
                    for warps in (8, 10, 16)
                }
            )
            == 3
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        for reader in readers:
            reader.close()
