# Blender 4.2+ reads blender_manifest.toml and ignores bl_info;
# bl_info keeps the package installable as a legacy add-on in Blender 3.6.
bl_info = {
    "name": "Mesh2GS",
    "author": "David Kadimyan",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar > Mesh2GS",
    "description": "Convert polygon models to Gaussian Splatting",
    "category": "Import-Export",
}

try:
    import bpy
except ModuleNotFoundError:
    bpy = None  # core/ and io/ are also imported outside Blender (tests, external trainer)

if bpy is not None:
    from . import i18n, setup_env, training
    from .operators import (MESH2GS_OT_cancel_install, MESH2GS_OT_cancel_training, MESH2GS_OT_check_python,
                            MESH2GS_OT_convert, MESH2GS_OT_copy_setup, MESH2GS_OT_export_ply,
                            MESH2GS_OT_export_spz, MESH2GS_OT_import_ply, MESH2GS_OT_import_spz,
                            MESH2GS_OT_install_env, MESH2GS_OT_scene_light, MESH2GS_OT_toggle_source,
                            _watch_install)
    from .props import Mesh2GSPreferences, Mesh2GSSettings
    from .ui import MENUS, PANELS, import_entry, menu_entry

    classes = (Mesh2GSPreferences, Mesh2GSSettings, MESH2GS_OT_convert, MESH2GS_OT_cancel_training,
               MESH2GS_OT_check_python, MESH2GS_OT_copy_setup, MESH2GS_OT_install_env,
               MESH2GS_OT_cancel_install, MESH2GS_OT_scene_light, MESH2GS_OT_toggle_source,
               MESH2GS_OT_export_ply, MESH2GS_OT_export_spz, MESH2GS_OT_import_ply, MESH2GS_OT_import_spz,
               *PANELS)

    def register():
        for cls in classes:
            bpy.utils.register_class(cls)
        i18n.apply(i18n.preference())  # after the preferences class exists
        bpy.types.Scene.mesh2gs = bpy.props.PointerProperty(type=Mesh2GSSettings)
        for menu in MENUS:
            getattr(bpy.types, menu).append(menu_entry)
        bpy.types.TOPBAR_MT_file_import.append(import_entry)

    def unregister():
        if training.ACTIVE is not None:  # stop a trainer that would outlive the add-on
            training.ACTIVE.cleanup()
            training.ACTIVE = None
        if setup_env.INSTALLER is not None and setup_env.INSTALLER.running:
            setup_env.INSTALLER.cancel()
        if bpy.app.timers.is_registered(_watch_install):
            bpy.app.timers.unregister(_watch_install)
        bpy.types.TOPBAR_MT_file_import.remove(import_entry)
        for menu in MENUS:
            getattr(bpy.types, menu).remove(menu_entry)
        del bpy.types.Scene.mesh2gs
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
        i18n.unregister()
