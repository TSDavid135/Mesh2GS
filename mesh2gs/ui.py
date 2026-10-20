import bpy
import numpy as np

from . import i18n, materials, preview, training
from .operators import (MESH2GS_OT_cancel_training, MESH2GS_OT_convert, MESH2GS_OT_export_ply,
                        MESH2GS_OT_export_spz, MESH2GS_OT_import_ply, MESH2GS_OT_import_spz,
                        MESH2GS_OT_scene_light, MESH2GS_OT_toggle_source, selected_sources, source_objects)
from .props import AVAILABLE_METHODS, BUDGET_METHODS, BYTES_PER_GAUSSIAN, TRAINED_METHODS

tr = i18n.tr
CTX = i18n.CTX


def _spaced(n):
    return f"{int(n):,}".replace(",", " ")


def _summary(objects):
    """Triangles and approximate world area of the original meshes (cheap enough for draw)."""
    tris, area = 0, 0.0
    for obj in objects:
        polys = obj.data.polygons
        loop_totals = np.empty(len(polys), dtype=np.int32)
        polys.foreach_get("loop_total", loop_totals)
        areas = np.empty(len(polys), dtype=np.float32)
        polys.foreach_get("area", areas)
        tris += int(loop_totals.sum()) - 2 * len(polys)
        scale = abs(np.linalg.det(np.array(obj.matrix_world, dtype=np.float64)[:3, :3])) ** (2 / 3)
        area += float(areas.sum()) * scale
    return tris, area


MAX_MATERIAL_ROWS = 5


def _draw_materials(col, objects, bake_resolution, lit=False, emission=True):
    """Where each material's color comes from: texture, constant or a fallback."""
    if lit:
        col.label(text=tr("All materials: scene light → bake {}²").format(bake_resolution), icon='LIGHT_SUN',
                  translate=False)
        return
    seen = {}
    for obj in objects:
        for slot in obj.material_slots or [None]:
            mat = slot.material if slot else None
            seen.setdefault(mat.name if mat else "", mat)
    for i, (name, mat) in enumerate(seen.items()):
        if i == MAX_MATERIAL_ROWS:
            col.label(text=tr("… {} more materials").format(len(seen) - MAX_MATERIAL_ROWS), translate=False)
            break
        info = materials.slot_color(mat)
        label = name or tr("no material")
        if info.emissive and emission and bake_resolution:
            text, icon = tr("{}: color + glow → bake {}²").format(label, bake_resolution), 'LIGHT'
        elif info.kind == 'TEXTURE':
            w, h = info.image.size
            text, icon = f"{label}: {info.image.name} {w}×{h}", 'TEXTURE'
        elif info.kind == 'CONSTANT':
            text, icon = tr("{}: color").format(label), 'COLOR'
        elif info.bakeable and bake_resolution:
            text, icon = tr("{}: {} → bake {}²").format(label, info.note, bake_resolution), 'RENDER_STILL'
        else:
            text, icon = tr("{}: {}, viewport color").format(label, info.note), 'ERROR'
        col.label(text_ctxt=CTX, text=text, icon=icon, translate=False)


def draw_settings(layout, context):
    """Conversion settings; shared by the sidebar panel and the popup dialog."""
    s = context.scene.mesh2gs
    sources = selected_sources(context)
    tris, area = _summary(sources)

    box = layout.box()
    col = box.column(align=True)
    if sources:
        col.label(text=tr("Objects: {}").format(len(sources)), icon='OBJECT_DATA', translate=False)
        col.label(text=tr("Triangles: {}").format(_spaced(tris)), icon='MESH_DATA', translate=False)
        col.label(text=tr("Area: {:.2f} m²").format(area), icon='SURFACE_NSURFACE', translate=False)
        lit = s.color_mode == 'LIT'
        _draw_materials(box.column(align=True), sources,
                        int(s.bake_resolution) if (s.bake_procedural or lit) else 0, lit, s.use_emission)
    else:
        col.label(text_ctxt=CTX, text="Select mesh objects", icon='INFO')
    if context.mode == 'EDIT_MESH':
        box.prop(s, "selected_faces_only")

    layout.prop(s, "method")
    if s.method not in AVAILABLE_METHODS:
        layout.label(text_ctxt=CTX, text="This method comes in a later version", icon='ERROR')
        return
    layout.row().prop(s, "color_mode", expand=True)
    if s.color_mode == 'LIT':
        col = layout.box().column(align=True)
        col.label(text_ctxt=CTX, text="Light from a splat scene (optional):", icon='WORLD')
        col.prop(s, "light_scene", text="")
        col.operator(MESH2GS_OT_scene_light.bl_idname, icon='IMAGE_BACKGROUND')

    if s.method in BUDGET_METHODS:
        layout.row().prop(s, "budget_mode", expand=True)
        if s.budget_mode == 'COUNT':
            layout.prop(s, "budget")
            estimate = s.budget
        else:
            layout.prop(s, "density")
            estimate = s.density * area
    else:
        layout.row().prop(s, "per_triangle", expand=True)
        estimate = tris * int(s.per_triangle)
    if s.method in TRAINED_METHODS:
        layout.prop(s, "m3_iterations" if s.method == 'M3' else "m2_iterations")

    size_mb = estimate * BYTES_PER_GAUSSIAN / 1e6
    layout.label(text=tr("≈ {} Gaussians · {:.1f} MB PLY").format(_spaced(estimate), size_mb),
                 icon='POINTCLOUD_DATA', translate=False)
    if s.method in TRAINED_METHODS:
        if training.python_path(context):
            layout.label(text=tr("Training: NVIDIA GPU, {} views").format(s.train_views), icon='INFO',
                         translate=False)
        else:
            col = layout.column(align=True)
            col.label(text_ctxt=CTX, text="Needs PyTorch and gsplat (one button, ≈5 GB):", icon='ERROR')
            op = col.operator("preferences.addon_show", text_ctxt=CTX, text="Install components…", icon='PREFERENCES')
            op.module = __package__


def draw_progress(layout, job):
    box = layout.box()
    col = box.column(align=True)
    if hasattr(col, "progress"):  # Blender 4.0+
        col.progress(factor=job.fraction, type='BAR', text=job.text)
    else:
        col.label(text_ctxt=CTX, text=f"{job.text} ({job.fraction:.0%})", icon='TIME', translate=False)
    col.label(text_ctxt=CTX, text="1/2 rendering views (Esc cancels)" if job.stage == "VIEWS"
              else "2/2 training: you can keep working")
    col.label(text=tr("{:.0f} s so far").format(job.seconds), translate=False)
    box.operator(MESH2GS_OT_cancel_training.bl_idname, icon='CANCEL')


class VIEW3D_PT_mesh2gs(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_translation_context = CTX
    bl_region_type = 'UI'
    bl_category = "Mesh2GS"
    bl_label = "Mesh → Gaussian Splatting"

    def draw(self, context):
        layout = self.layout
        job = training.ACTIVE
        if job is not None and job.running:
            draw_progress(layout, job)
            return
        draw_settings(layout, context)
        row = layout.row()
        row.scale_y = 1.6
        row.operator_context = 'EXEC_DEFAULT'  # the panel already shows the settings
        row.operator(MESH2GS_OT_convert.bl_idname, text_ctxt=CTX, text="Convert", icon='POINTCLOUD_DATA')


class VIEW3D_PT_mesh2gs_advanced(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_translation_context = CTX
    bl_region_type = 'UI'
    bl_category = "Mesh2GS"
    bl_parent_id = "VIEW3D_PT_mesh2gs"
    bl_label = "Advanced"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.mesh2gs
        col = self.layout.column()
        if s.method in TRAINED_METHODS:
            col.prop(s, "train_views")
            col.prop(s, "train_size")
        if s.method == 'M3':
            col.label(text_ctxt=CTX, text="M3 starts from M1 with these settings:")
        if s.method in {'M1', 'M3'}:
            col.prop(s, "sampling")
            row = col.row()
            row.prop(s, "edge_aware")
            sub = row.row()
            sub.active = s.edge_aware
            sub.prop(s, "sharp_angle", text="")
            col.prop(s, "overlap")
            col.prop(s, "seed")
        row = col.row()
        row.prop(s, "bake_procedural")
        sub = row.row()
        sub.active = s.bake_procedural or s.color_mode == 'LIT'
        sub.prop(s, "bake_resolution", text="")
        if s.color_mode == 'ALBEDO':
            sub = col.row()
            sub.active = s.bake_procedural
            sub.prop(s, "use_emission")
        if s.color_mode == 'LIT':
            col.prop(s, "lit_samples")
        col.prop(s, "preview_limit")
        col.prop(s, "hide_source")


class VIEW3D_PT_mesh2gs_result(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_translation_context = CTX
    bl_region_type = 'UI'
    bl_category = "Mesh2GS"
    bl_label = "Result"

    @classmethod
    def poll(cls, context):
        return preview.is_gaussian_object(context.active_object)

    def draw(self, context):
        obj = context.active_object
        count = obj.get("mesh2gs_count", len(obj.data.vertices))
        layout = self.layout
        col = layout.box().column(align=True)
        col.label(text_ctxt=CTX, text=obj.name, icon='POINTCLOUD_DATA', translate=False)
        col.label(text=tr("Method: {}").format(obj[preview.TAG]), translate=False)
        col.label(text=tr("Gaussians: {}").format(_spaced(count)), translate=False)
        col.label(text=tr("Time: {:.2f} s").format(obj.get("mesh2gs_seconds", 0.0)), translate=False)
        col.label(text=tr("PLY: {:.1f} MB").format(count * BYTES_PER_GAUSSIAN / 1e6), translate=False)
        sources = source_objects(obj)
        if sources:
            layout.operator(MESH2GS_OT_toggle_source.bl_idname,
                            icon='HIDE_OFF' if obj.hide_get() else 'HIDE_ON')
        row = layout.row()
        row.scale_y = 1.4
        row.operator(MESH2GS_OT_export_ply.bl_idname, text_ctxt=CTX, text="Export PLY…", icon='EXPORT')
        row.operator(MESH2GS_OT_export_spz.bl_idname, text_ctxt=CTX, text="Export SPZ…", icon='EXPORT')


def menu_entry(self, context):
    self.layout.separator()
    self.layout.operator_context = 'INVOKE_DEFAULT'
    self.layout.operator(MESH2GS_OT_convert.bl_idname, text_ctxt=CTX, text="Mesh2GS: to Gaussians…", icon='POINTCLOUD_DATA')


def import_entry(self, context):
    self.layout.operator(MESH2GS_OT_import_ply.bl_idname, text_ctxt=CTX, text="Gaussian Splatting (.ply, Mesh2GS)")
    self.layout.operator(MESH2GS_OT_import_spz.bl_idname, text_ctxt=CTX, text="Gaussian Splatting (.spz, Mesh2GS)")


MENUS = ("VIEW3D_MT_object", "VIEW3D_MT_object_context_menu", "VIEW3D_MT_edit_mesh_context_menu")
PANELS = (VIEW3D_PT_mesh2gs, VIEW3D_PT_mesh2gs_advanced, VIEW3D_PT_mesh2gs_result)
