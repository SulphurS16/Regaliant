//! Isolated scatter diagnostic. Uses locally captured code/constants, never guest execution.
//! Controlled depth + zero media + NaN history: valid slices must stay finite, excluded
//! slices must contain the shader's NaN sentinel. No shader/pipeline cache is consulted.
use ash::vk;
use rdna::translate::{self, BindingKind as B, ImageDim as D, NumClass, ResSpec, Spec};
use std::{io::Write, path::Path};
fn words(p: &Path) -> Result<Vec<u32>, String> {
    Ok(std::fs::read(p)
        .map_err(|e| e.to_string())?
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
pub fn run(input: &Path, output: &Path) -> Result<(), String> {
    let output = crate::platform::files::output_path(output, &[]).map_err(err)?;
    std::fs::create_dir_all(&output).map_err(err)?;
    crate::log::init(&output.join("dsport.log")).map_err(err)?;
    unsafe {
        std::env::set_var("DS_LINUX_HIDDEN", "1");
    }
    super::vk::init()?;
    let c = super::vk::ctx();
    let d = &c.device;
    let raw_code = words(&input.join("shader.bin"))?;
    let code = rdna::decode_program(&raw_code).map_err(|e| e.0)?;
    let mut spec = Spec::from_words(&std::fs::read_to_string(input.join("spec.txt")).map_err(err)?).ok_or("bad spec")?;
    if spec.stage != translate::Stage::Compute || spec.cs_local_size != [8, 8, 1] {
        return Err("expected captured scatter compute [8,8,1]".into());
    }
    let hash = super::shaders::hash_words(&raw_code);
    super::shaders::note_program(&raw_code, &code, hash);
    super::shaders::apply_modes(&mut spec, hash);
    // Preserve control flow and formats; replace the host's address/ring machinery with
    // explicit standalone descriptors. This tests translation separately from guest binding.
    spec.addresses = false;
    spec.sgpr_pc = None;
    spec.sgpr_ubo = 0;
    spec.batched = false;
    spec.write_marks = false;
    spec.gpu_loads = 0;
    spec.gpu_res = 0;
    spec.lod_stats = false;
    let plan = rdna::analysis::analyze(&code, spec.user_sgprs, spec.user_base);
    let mut sgpr = words(&input.join("sgpr.bin"))?;
    let off = translate::load_offsets(&plan);
    let coeff = *plan.inst_load.get(&0x28).ok_or("missing depth coefficients")?;
    let base_slice = *plan.inst_load.get(&0x30).ok_or("missing slice offset")?;
    sgpr[off[base_slice] as usize] = 0;
    // depth=1, log2(1)=0: the limit is exactly the last coefficient (0,4,71).
    let co = off[coeff] as usize;
    sgpr[co..co + 4].copy_from_slice(&[1f32.to_bits(), 0, 1f32.to_bits(), 0]);
    let mut report = std::fs::File::create(output.join("summary.txt")).map_err(err)?;
    writeln!(
        report,
        "Isolated full scatter: 8x8x72, depth=1, zero media, NaN history. Dummy zero resources for unreachable lighting. subgroup={} requested32={:?}; per_lane={} lane_votes={}",
        c.subgroup_size, c.subgroup32, spec.per_lane, spec.lane_votes
    )
    .map_err(err)?;
    writeln!(report, "Probes save one register after a PC; execution continues. Unreached probes default to zero. Read slices.csv values and history-instructions.txt, not just finite counts.").map_err(err)?;
    std::fs::copy(input.join("history-instructions.txt"), output.join("history-instructions.txt")).map_err(err)?;
    let probes = std::fs::read_to_string(input.join("probes.txt")).map_err(err)?;
    let original_spec = spec.clone();
    let mut slice_report = std::fs::File::create(output.join("slices.csv")).map_err(err)?;
    writeln!(slice_report, "variant,limit,slice,finite_columns,first_rgba_bits").map_err(err)?;
    let mut failed = false;
    let mut baselines = 0;
    for line in probes.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 3 || !fields[0].bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err(format!("invalid probe manifest line {line:?}"));
        }
        let name = fields[0];
        let probe = if fields[1] == "-" && fields[2] == "-" {
            None
        } else {
            Some((fields[1].parse::<u32>().map_err(err)?, fields[2].parse::<u32>().map_err(err)?))
        };
        if name == "normal" || name == "normal-half" {
            baselines += 1;
        }
        let half = name == "normal-half";
        spec = original_spec.clone();
        spec.probe = probe;
        if name == "no-lane-votes" {
            spec.lane_votes = false;
            spec.known_active = false;
        }
        if name == "no-lane-guards" {
            spec.lane_guards = false;
        }
        writeln!(
            report,
            "variant {name}: per_lane={} lane_votes={} lane_guards={}",
            spec.per_lane, spec.lane_votes, spec.lane_guards
        )
        .map_err(err)?;
        report.flush().map_err(err)?;
        let tr = translate::translate(&code, &plan, &spec)?;
        let module = super::exec::module_code(&tr.spirv);
        std::fs::write(output.join(format!("{name}.spv")), module.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>()).map_err(err)?;
        unsafe {
            let pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(c.queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(err)?;
            let cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(err)?[0];
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(err)?;
            let mut bufs = Vec::new();
            let mut imgs = Vec::new();
            let mut views = Vec::new();
            let mut samplers = Vec::new();
            let mut bindings = Vec::new();
            let mut bis = Vec::new();
            let mut iis = Vec::new();
            let mut types = Vec::new();
            let mut sgptr = std::ptr::null_mut();
            let mut watchdog = std::ptr::null_mut();
            let mut out_image = None;
            for b in &tr.bindings {
                let (ty, bi, ii) = match b.kind {
                    B::SgprData | B::Buffer(_) | B::Watchdog | B::Gds => {
                        let size = if b.kind == B::SgprData { sgpr.len() * 4 } else { 1 << 20 };
                        let (buf, mem, ptr) = c.host_buffer(size as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
                        std::ptr::write_bytes(ptr, 0, size);
                        if b.kind == B::SgprData {
                            sgptr = ptr;
                        }
                        if b.kind == B::Watchdog {
                            watchdog = ptr;
                        }
                        bufs.push((buf, mem));
                        (
                            vk::DescriptorType::STORAGE_BUFFER,
                            vk::DescriptorBufferInfo::default().buffer(buf).range(size as u64),
                            vk::DescriptorImageInfo::default(),
                        )
                    }
                    B::Sampler(_) | B::CompareSampler(_) => {
                        let sampler = d
                            .create_sampler(
                                &vk::SamplerCreateInfo::default()
                                    .mag_filter(vk::Filter::LINEAR)
                                    .min_filter(vk::Filter::LINEAR)
                                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                                    .compare_enable(matches!(b.kind, B::CompareSampler(_)))
                                    .compare_op(vk::CompareOp::LESS_OR_EQUAL),
                                None,
                            )
                            .map_err(err)?;
                        samplers.push(sampler);
                        (
                            vk::DescriptorType::SAMPLER,
                            vk::DescriptorBufferInfo::default(),
                            vk::DescriptorImageInfo::default().sampler(sampler),
                        )
                    }
                    B::Image(i) | B::ImageAlt(i) | B::StorageImage(i) => {
                        let ResSpec::Image { dim, class, .. } = spec.resources[i] else {
                            return Err("expected image spec".into());
                        };
                        let dim = if matches!(b.kind, B::ImageAlt(_)) { D::D2 } else { dim };
                        let (ity, vty, layers, depth, flags) = match dim {
                            D::D2 => (vk::ImageType::TYPE_2D, vk::ImageViewType::TYPE_2D, 1, 1, vk::ImageCreateFlags::empty()),
                            D::D3 | D::D3OrD2 => (vk::ImageType::TYPE_3D, vk::ImageViewType::TYPE_3D, 1, 72, vk::ImageCreateFlags::empty()),
                            D::D2Array => (vk::ImageType::TYPE_2D, vk::ImageViewType::TYPE_2D_ARRAY, 72, 1, vk::ImageCreateFlags::empty()),
                            D::Cube | D::CubeArray => (
                                vk::ImageType::TYPE_2D,
                                if dim == D::Cube { vk::ImageViewType::CUBE } else { vk::ImageViewType::CUBE_ARRAY },
                                6,
                                1,
                                vk::ImageCreateFlags::CUBE_COMPATIBLE,
                            ),
                            _ => return Err(format!("unsupported fixture dimension {dim:?}")),
                        };
                        let format = if half && i == 0 {
                            vk::Format::R16_SFLOAT
                        } else if half && matches!(i, 1 | 2 | 3) {
                            vk::Format::R16G16B16A16_SFLOAT
                        } else {
                            match class {
                                NumClass::Float => vk::Format::R32G32B32A32_SFLOAT,
                                NumClass::Uint => vk::Format::R32G32B32A32_UINT,
                                NumClass::Sint => vk::Format::R32G32B32A32_SINT,
                            }
                        };
                        let image = d
                            .create_image(
                                &vk::ImageCreateInfo::default()
                                    .flags(flags)
                                    .image_type(ity)
                                    .format(format)
                                    .extent(vk::Extent3D { width: 8, height: 8, depth })
                                    .mip_levels(1)
                                    .array_layers(layers)
                                    .samples(vk::SampleCountFlags::TYPE_1)
                                    .tiling(vk::ImageTiling::OPTIMAL)
                                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST),
                                None,
                            )
                            .map_err(err)?;
                        let req = d.get_image_memory_requirements(image);
                        let mt = (0..c.mem_props.memory_type_count)
                            .find(|&t| req.memory_type_bits & (1 << t) != 0 && c.mem_props.memory_types[t as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
                            .ok_or("image memory type")?;
                        let mem = d
                            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt), None)
                            .map_err(err)?;
                        d.bind_image_memory(image, mem, 0).map_err(err)?;
                        let range = vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(layers);
                        let view = d
                            .create_image_view(
                                &vk::ImageViewCreateInfo::default().image(image).view_type(vty).format(format).subresource_range(range),
                                None,
                            )
                            .map_err(err)?;
                        imgs.push((image, mem, range, i));
                        views.push(view);
                        if i == 1 {
                            out_image = Some(image);
                        }
                        (
                            if matches!(b.kind, B::StorageImage(_)) {
                                vk::DescriptorType::STORAGE_IMAGE
                            } else {
                                vk::DescriptorType::SAMPLED_IMAGE
                            },
                            vk::DescriptorBufferInfo::default(),
                            vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL),
                        )
                    }
                    _ => return Err(format!("unsupported fixture binding {:?}", b.kind)),
                };
                types.push(ty);
                bis.push(bi);
                iis.push(ii);
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b.binding)
                        .descriptor_type(ty)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            let sizes = types.iter().map(|&t| vk::DescriptorPoolSize { ty: t, descriptor_count: 1 }).collect::<Vec<_>>();
            let dp = d
                .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)
                .map_err(err)?;
            let dl = d
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
                .map_err(err)?;
            let set = d
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(dp).set_layouts(&[dl]))
                .map_err(err)?[0];
            let writes = tr
                .bindings
                .iter()
                .enumerate()
                .map(|(j, b)| {
                    let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b.binding).descriptor_type(types[j]);
                    if types[j] == vk::DescriptorType::STORAGE_BUFFER {
                        w.buffer_info(&bis[j..j + 1])
                    } else {
                        w.image_info(&iis[j..j + 1])
                    }
                })
                .collect::<Vec<_>>();
            d.update_descriptor_sets(&writes, &[]);
            let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[dl]), None).map_err(err)?;
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&module), None).map_err(err)?;
            let mut sg = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let stage = if c.subgroup32.contains(vk::ShaderStageFlags::COMPUTE) {
                stage.push_next(&mut sg)
            } else {
                stage
            };
            let pipeline = d
                .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None)
                .map_err(err)?[0];
            let size = 8 * 8 * 72 * if half { 8 } else { 16 };
            let (read, readmem, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
            for (iteration, limit) in [0u32, 4, 71].into_iter().enumerate() {
                sgpr[co + 3] = (limit as f32).to_bits();
                assert!(!sgptr.is_null());
                std::ptr::copy_nonoverlapping(sgpr.as_ptr().cast::<u8>(), sgptr, sgpr.len() * 4);
                d.reset_command_pool(pool, vk::CommandPoolResetFlags::empty()).map_err(err)?;
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(err)?;
                for &(image, _, range, i) in &imgs {
                    let barrier = vk::ImageMemoryBarrier::default()
                        .image(image)
                        .subresource_range(range)
                        .old_layout(if iteration == 0 { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL })
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_access_mask(if iteration == 0 {
                            vk::AccessFlags::empty()
                        } else {
                            vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ
                        })
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                    d.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                    let clear = vk::ClearColorValue {
                        float32: if i == 0 {
                            [1.; 4]
                        } else if i == 3 {
                            [f32::from_bits(u32::MAX); 4]
                        } else if i == 1 {
                            [-8192.; 4]
                        } else {
                            [0.; 4]
                        },
                    };
                    d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &clear, &[range]);
                }
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                d.cmd_dispatch(cmd, 1, 1, 72);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                    &[],
                    &[],
                );
                d.cmd_copy_image_to_buffer(
                    cmd,
                    out_image.ok_or("no scatter output")?,
                    vk::ImageLayout::GENERAL,
                    read,
                    &[vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                        .image_extent(vk::Extent3D { width: 8, height: 8, depth: 72 })],
                );
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)],
                    &[],
                    &[],
                );
                d.end_command_buffer(cmd).map_err(err)?;
                c.submit(cmd, &[], &[], fence)?;
                d.wait_for_fences(&[fence], true, 60_000_000_000).map_err(err)?;
                d.reset_fences(&[fence]).map_err(err)?;
                if !watchdog.is_null() {
                    let bytes = std::slice::from_raw_parts(watchdog, translate::WATCHDOG_DWORDS as usize * 4);
                    std::fs::write(output.join(format!("{name}-limit{limit}-watchdog.bin")), bytes).map_err(err)?;
                    let trips = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                    writeln!(report, "{name} limit {limit}: cumulative watchdog trips {trips}").map_err(err)?;
                    if probe.is_none() && trips != 0 {
                        failed = true;
                    }
                }
                let raw = std::slice::from_raw_parts(ptr, size as usize);
                std::fs::write(output.join(format!("{name}-limit{limit}.{}", if half { "rgba16f" } else { "rgba32f" })), raw).map_err(err)?;
                let vals: Vec<_> = if half {
                    raw.chunks_exact(2).map(|b| super::images::half_to_f32(u16::from_le_bytes(b.try_into().unwrap()))).collect()
                } else {
                    raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
                };
                let counts: Vec<_> = (0..64)
                    .map(|xy| (0..72).filter(|&z| vals[(z * 64 + xy) * 4..(z * 64 + xy) * 4 + 4].iter().all(|v| v.is_finite())).count())
                    .collect();
                for (z, slice) in vals.chunks_exact(64 * 4).enumerate() {
                    let finite = slice.chunks_exact(4).filter(|p| p.iter().all(|v| v.is_finite())).count();
                    writeln!(
                        slice_report,
                        "{name},{limit},{z},{finite},{:08x} {:08x} {:08x} {:08x}",
                        slice[0].to_bits(),
                        slice[1].to_bits(),
                        slice[2].to_bits(),
                        slice[3].to_bits()
                    )
                    .map_err(err)?;
                }
                slice_report.flush().map_err(err)?;
                if name == "history-accept-mask" || name == "history-combined-mask" || name == "history-pc-444c" {
                    // NaN history must be rejected in every lane, not merely produce a
                    // finite 0/1 mask. The old classifier returned 1 in valid slices.
                    if vals.iter().any(|v| *v != 0.0) {
                        writeln!(report, "FAILED: {name} accepted NaN history at limit {limit}").map_err(err)?;
                        failed = true;
                    }
                }
                if name == "normal" || half {
                    for (z, slice) in vals.chunks_exact(64 * 4).enumerate() {
                        if !slice.iter().all(|v| if z <= limit as usize { *v == 0.0 } else { v.is_nan() }) {
                            failed = true;
                        }
                    }
                }
                let rgba: Vec<u8> = vals
                    .chunks_exact(4)
                    .flat_map(|p| {
                        if p.iter().all(|v| v.is_nan()) {
                            [255, 0, 0, 255]
                        } else if p.iter().all(|v| *v == -8192.0) {
                            [255, 0, 255, 255]
                        } else if p.iter().all(|v| v.is_finite()) {
                            [0, 255, 0, 255]
                        } else {
                            [255, 255, 0, 255]
                        }
                    })
                    .collect();
                std::fs::write(output.join(format!("{name}-limit{limit}.png")), super::png::encode(64, 72, &rgba)).map_err(err)?;
                writeln!(report, "{name} limit {limit}: finite slices per column {counts:?}; expected normal {}", limit + 1).map_err(err)?;
                report.flush().map_err(err)?;
            }
            d.destroy_buffer(read, None);
            d.free_memory(readmem, None);
            d.destroy_pipeline(pipeline, None);
            d.destroy_shader_module(module, None);
            d.destroy_pipeline_layout(layout, None);
            d.destroy_descriptor_pool(dp, None);
            d.destroy_descriptor_set_layout(dl, None);
            for v in views {
                d.destroy_image_view(v, None);
            }
            for (i, m, _, _) in imgs {
                d.destroy_image(i, None);
                d.free_memory(m, None);
            }
            for (b, m) in bufs {
                d.destroy_buffer(b, None);
                d.free_memory(m, None);
            }
            for s in samplers {
                d.destroy_sampler(s, None);
            }
            d.destroy_fence(fence, None);
            d.destroy_command_pool(pool, None);
        }
    }
    writeln!(
        report,
        "PNG: x=flattened 8x8 column, y=slice. Green finite, red all NaN, magenta unwritten, yellow mixed. Normal checks: {}",
        if failed { "FAILED" } else { "PASSED" }
    )
    .map_err(err)?;
    crate::log::flush();
    if baselines != 2 {
        return Err("probe manifest must include normal and normal-half".into());
    }
    if failed {
        Err("isolated scatter finite-slice check failed; inspect summary.txt and probe outputs".into())
    } else {
        Ok(())
    }
}
