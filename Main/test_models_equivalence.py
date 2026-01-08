import copy
import importlib.util
import os
import unittest

import torch


def _load_bak_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bak_path = os.path.join(repo_root, "Main", "models.bak.py")
    spec = importlib.util.spec_from_file_location("models_bak", bak_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _signature(model: torch.nn.Module):
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()}


def _count_parameters(model: torch.nn.Module):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _parameter_breakdown(model: torch.nn.Module):
    """Get detailed parameter counts by layer name."""
    return {
        name: {"shape": tuple(p.shape), "count": p.numel()}
        for name, p in model.named_parameters()
    }


# ============================================================================
# State Dict Mapping Functions
# ============================================================================

def map_deeponet_darcy_state_dict(new_state_dict):
    """Map new DeepONet_darcy state dict keys to old format."""
    old_state_dict = {}
    for key, value in new_state_dict.items():
        if key.startswith("branch.0."):
            # branch.0.N.* -> branch.N.*
            old_key = key.replace("branch.0.", "branch.")
            old_state_dict[old_key] = value
        elif key.startswith("trunk.0."):
            # trunk.0.N.* -> trunk.N.*
            old_key = key.replace("trunk.0.", "trunk.")
            old_state_dict[old_key] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def map_deeponet_plate_state_dict(new_state_dict):
    """Map new DeepONet_plate state dict keys to old format."""
    old_state_dict = {}
    for key, value in new_state_dict.items():
        if key.startswith("branch.0."):
            # branch.0.N.* -> branch1.N.*
            old_key = key.replace("branch.0.", "branch1.")
            old_state_dict[old_key] = value
        elif key.startswith("branch.1."):
            # branch.1.N.* -> branch2.N.*
            old_key = key.replace("branch.1.", "branch2.")
            old_state_dict[old_key] = value
        elif key.startswith("trunk.0."):
            # trunk.0.N.* -> trunk1.N.*
            old_key = key.replace("trunk.0.", "trunk1.")
            old_state_dict[old_key] = value
        elif key.startswith("trunk.1."):
            # trunk.1.N.* -> trunk2.N.*
            old_key = key.replace("trunk.1.", "trunk2.")
            old_state_dict[old_key] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def map_dcon_darcy_state_dict(new_state_dict):
    """Map new DCON_darcy state dict keys to old format."""
    old_state_dict = {}
    for key, value in new_state_dict.items():
        if key.startswith("FC.0.0."):
            # FC.0.0.* -> FC1u.* (ModuleList[0] -> ModuleList[0])
            old_key = key.replace("FC.0.0.", "FC1u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.1."):
            # FC.0.1.* -> FC2u.*
            old_key = key.replace("FC.0.1.", "FC2u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.2."):
            # FC.0.2.* -> FC3u.*
            old_key = key.replace("FC.0.2.", "FC3u.")
            old_state_dict[old_key] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def map_dcon_plate_state_dict(new_state_dict):
    """Map new DCON_plate state dict keys to old format."""
    old_state_dict = {}
    for key, value in new_state_dict.items():
        # u-component (field_dim 0)
        if key.startswith("FC.0.0."):
            # FC.0.0.* -> FC1u.* (ModuleList[0] -> ModuleList[0])
            old_key = key.replace("FC.0.0.", "FC1u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.1."):
            # FC.0.1.* -> FC2u.*
            old_key = key.replace("FC.0.1.", "FC2u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.2."):
            # FC.0.2.* -> FC3u.*
            old_key = key.replace("FC.0.2.", "FC3u.")
            old_state_dict[old_key] = value
        # v-component (field_dim 1)
        elif key.startswith("FC.1.0."):
            # FC.1.0.* -> FC1v.*
            old_key = key.replace("FC.1.0.", "FC1v.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.1."):
            # FC.1.1.* -> FC2v.*
            old_key = key.replace("FC.1.1.", "FC2v.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.2."):
            # FC.1.2.* -> FC3v.*
            old_key = key.replace("FC.1.2.", "FC3v.")
            old_state_dict[old_key] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def map_improved_deeponet_darcy_state_dict(new_state_dict):
    """Map new Improved_DeepOnet_darcy state dict keys to old format.

    New: FC.0.0.*, FC.0.1.*, FC.0.2.* (ModuleList of trunk layers)
    Old: FC1c.*, FC2c.*, FC3c.* (direct attributes)
    """
    old_state_dict = {}
    for key, value in new_state_dict.items():
        if key.startswith("FC.0.0."):
            old_key = key.replace("FC.0.0.", "FC1c.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.1."):
            old_key = key.replace("FC.0.1.", "FC2c.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.2."):
            old_key = key.replace("FC.0.2.", "FC3c.")
            old_state_dict[old_key] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def map_improved_deeponet_plate_state_dict(new_state_dict):
    """Map new Improved_DeepONet_plate state dict keys to old format.

    New: FC.0.0.*, FC.0.1.*, FC.0.2.*, FC.1.0.*, FC.1.1.*, FC.1.2.*
    Old: FC1c1.*, FC2c1.*, FC3c1.*, FC1c2.*, FC2c2.*, FC3c2.*
    """
    old_state_dict = {}
    for key, value in new_state_dict.items():
        if key.startswith("FC.0.0."):
            old_key = key.replace("FC.0.0.", "FC1c1.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.1."):
            old_key = key.replace("FC.0.1.", "FC2c1.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.2."):
            old_key = key.replace("FC.0.2.", "FC3c1.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.0."):
            old_key = key.replace("FC.1.0.", "FC1c2.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.1."):
            old_key = key.replace("FC.1.1.", "FC2c2.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.2."):
            old_key = key.replace("FC.1.2.", "FC3c2.")
            old_state_dict[old_key] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def filter_dead_code_state_dict(state_dict, dead_layers=None):
    """Remove dead code layers from state dict for comparison.

    Args:
        state_dict: The model state dict to filter
        dead_layers: List of layer name prefixes to exclude.
                     Default includes DCON_plate's lift/val_lift and DCON_darcy's FC4u.

    Returns:
        Filtered state dict without dead code layers
    """
    if dead_layers is None:
        # Dead code layers that are defined but never used in forward pass:
        # - DCON_plate: lift, val_lift
        # - DCON_darcy: FC4u (marked as "unused?" in original code)
        dead_layers = ['lift', 'val_lift', 'FC4u']
    return {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(dl + '.') or k == dl for dl in dead_layers)
    }


def translate_state_dict_for_model(state_dict, model_type):
    """Translate new model state dict to old model format.

    Args:
        state_dict: State dict from new model
        model_type: Type of model ('deeponet_darcy', 'deeponet_plate',
                                      'dcon_darcy', 'dcon_plate',
                                      'improved_deeponet_darcy', 'improved_deeponet_plate')

    Returns:
        State dict with keys translated to old format
    """
    if model_type == 'deeponet_darcy':
        return map_deeponet_darcy_state_dict(state_dict)
    elif model_type == 'deeponet_plate':
        return map_deeponet_plate_state_dict(state_dict)
    elif model_type == 'improved_deeponet_darcy':
        return map_improved_deeponet_darcy_state_dict(state_dict)
    elif model_type == 'improved_deeponet_plate':
        return map_improved_deeponet_plate_state_dict(state_dict)
    elif model_type == 'dcon_darcy':
        return map_dcon_darcy_state_dict(state_dict)
    elif model_type == 'dcon_plate':
        return map_dcon_plate_state_dict(state_dict)
    else:
        return state_dict


# =========================================================================
# Reverse State Dict Mapping Functions (Old -> New)
# =========================================================================


def reverse_map_deeponet_darcy_state_dict(old_state_dict):
    """Map old DeepONet_darcy state dict keys to new format."""
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith("branch."):
            # branch.N.* -> branch.0.N.*
            new_key = key.replace("branch.", "branch.0.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("trunk."):
            # trunk.N.* -> trunk.0.N.*
            new_key = key.replace("trunk.", "trunk.0.", 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def reverse_map_deeponet_plate_state_dict(old_state_dict):
    """Map old DeepONet_plate state dict keys to new format."""
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith("branch1."):
            new_key = key.replace("branch1.", "branch.0.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("branch2."):
            new_key = key.replace("branch2.", "branch.1.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("trunk1."):
            new_key = key.replace("trunk1.", "trunk.0.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("trunk2."):
            new_key = key.replace("trunk2.", "trunk.1.", 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def reverse_map_dcon_darcy_state_dict(old_state_dict):
    """Map old DCON_darcy state dict keys to new format."""
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith("FC1u."):
            new_key = key.replace("FC1u.", "FC.0.0.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC2u."):
            new_key = key.replace("FC2u.", "FC.0.1.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC3u."):
            new_key = key.replace("FC3u.", "FC.0.2.", 1)
            new_state_dict[new_key] = value
        else:
            # Preserve keys we don't recognize; load_state_dict(strict=False) will ignore extras.
            new_state_dict[key] = value
    return new_state_dict


def reverse_map_dcon_plate_state_dict(old_state_dict):
    """Map old DCON_plate state dict keys to new format."""
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith("FC1u."):
            new_key = key.replace("FC1u.", "FC.0.0.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC2u."):
            new_key = key.replace("FC2u.", "FC.0.1.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC3u."):
            new_key = key.replace("FC3u.", "FC.0.2.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC1v."):
            new_key = key.replace("FC1v.", "FC.1.0.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC2v."):
            new_key = key.replace("FC2v.", "FC.1.1.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC3v."):
            new_key = key.replace("FC3v.", "FC.1.2.", 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def reverse_map_improved_deeponet_darcy_state_dict(old_state_dict):
    """Map old Improved_DeepOnet_darcy state dict keys to new format."""
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith("FC1c."):
            new_key = key.replace("FC1c.", "FC.0.0.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC2c."):
            new_key = key.replace("FC2c.", "FC.0.1.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC3c."):
            new_key = key.replace("FC3c.", "FC.0.2.", 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def reverse_map_improved_deeponet_plate_state_dict(old_state_dict):
    """Map old Improved_DeepONet_plate state dict keys to new format."""
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith("FC1c1."):
            new_key = key.replace("FC1c1.", "FC.0.0.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC2c1."):
            new_key = key.replace("FC2c1.", "FC.0.1.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC3c1."):
            new_key = key.replace("FC3c1.", "FC.0.2.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC1c2."):
            new_key = key.replace("FC1c2.", "FC.1.0.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC2c2."):
            new_key = key.replace("FC2c2.", "FC.1.1.", 1)
            new_state_dict[new_key] = value
        elif key.startswith("FC3c2."):
            new_key = key.replace("FC3c2.", "FC.1.2.", 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def reverse_translate_state_dict_for_model(state_dict, model_type):
    """Translate old model state dict to new model format."""
    if model_type == 'deeponet_darcy':
        return reverse_map_deeponet_darcy_state_dict(state_dict)
    elif model_type == 'deeponet_plate':
        return reverse_map_deeponet_plate_state_dict(state_dict)
    elif model_type == 'improved_deeponet_darcy':
        return reverse_map_improved_deeponet_darcy_state_dict(state_dict)
    elif model_type == 'improved_deeponet_plate':
        return reverse_map_improved_deeponet_plate_state_dict(state_dict)
    elif model_type == 'dcon_darcy':
        return reverse_map_dcon_darcy_state_dict(state_dict)
    elif model_type == 'dcon_plate':
        return reverse_map_dcon_plate_state_dict(state_dict)
    else:
        return state_dict


def _signature_from_dict(state_dict):
    """Create signature dict from state dict (for comparing translated state dicts)."""
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in state_dict.items()}


class TestModelsForwardEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure we import the new models from this folder.
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.repo_root = os.path.dirname(cls.this_dir)

        # Import new models
        import sys

        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        import models as models_new  # noqa: E402

        cls.models_new = models_new
        cls.models_bak = _load_bak_module()

    def _make_inputs(self, B=3, M=17, N=13, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        return x_coor, y_coor, par

    def _assert_close(self, a, b):
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)

    def _compare_forward_single_output(self, new_model, bak_model, x_coor, y_coor, par):
        # Determine model type for state dict translation
        model_type = self._get_model_type(new_model)

        # Translate new model state dict to old format for loading
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)

        # Filter dead code from old model's state dict for signature comparison
        bak_state_filtered = filter_dead_code_state_dict(bak_model.state_dict())
        bak_sig_filtered = _signature_from_dict(bak_state_filtered)
        translated_sig = _signature_from_dict(translated_state)

        self.assertEqual(translated_sig, bak_sig_filtered,
                         f"State dict signatures don't match after translation for {model_type}")

        # Load translated state dict into old model (strict=False to ignore extra keys like FC4u)
        bak_model.load_state_dict(translated_state, strict=False)

        new_model.eval()
        bak_model.eval()
        with torch.no_grad():
            out_new = new_model(x_coor, y_coor, par)
            out_bak = bak_model(x_coor, y_coor, par)
        self._assert_close(out_new, out_bak)

    def _compare_forward_two_outputs(self, new_model, bak_model, x_coor, y_coor, par):
        # Determine model type for state dict translation
        model_type = self._get_model_type(new_model)

        # Translate new model state dict to old format for loading
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)

        # Filter dead code from old model's state dict for signature comparison
        bak_state_filtered = filter_dead_code_state_dict(bak_model.state_dict())
        bak_sig_filtered = _signature_from_dict(bak_state_filtered)
        translated_sig = _signature_from_dict(translated_state)

        self.assertEqual(translated_sig, bak_sig_filtered,
                         f"State dict signatures don't match after translation for {model_type}")

        # Load translated state dict into old model (strict=False to ignore extra keys)
        bak_model.load_state_dict(translated_state, strict=False)

        new_model.eval()
        bak_model.eval()
        with torch.no_grad():
            out_new = new_model(x_coor, y_coor, par)
            out_bak = bak_model(x_coor, y_coor, par)

        self.assertIsInstance(out_new, tuple)
        self.assertIsInstance(out_bak, tuple)
        self.assertEqual(len(out_new), 2)
        self.assertEqual(len(out_bak), 2)
        self._assert_close(out_new[0], out_bak[0])
        self._assert_close(out_new[1], out_bak[1])

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'DeepONet_darcy': 'deeponet_darcy',
            'DeepONet_plate': 'deeponet_plate',
            'Improved_DeepOnet_darcy': 'improved_deeponet_darcy',
            'Improved_DeepONet_plate': 'improved_deeponet_plate',
            'DCON_darcy': 'dcon_darcy',
            'DCON_plate': 'dcon_plate',
        }
        return mapping.get(class_name, None)

    def test_deeponet_darcy_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)

        new = self.models_new.DeepONet_darcy(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_darcy(config, input_dim=13)
        self._compare_forward_single_output(new, bak, x_coor, y_coor, par)

    def test_improved_deeponet_darcy_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)

        new = self.models_new.Improved_DeepOnet_darcy(config, input_dim=13)
        bak = self.models_bak.Improved_DeepOnet_darcy(config, input_dim=13)
        self._compare_forward_single_output(new, bak, x_coor, y_coor, par)

    def test_dcon_darcy_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)

        new = self.models_new.DCON_darcy(config)
        bak = self.models_bak.DCON_darcy(config)
        self._compare_forward_single_output(new, bak, x_coor, y_coor, par)

    def test_new_model_darcy_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)

        new = self.models_new.New_model_darcy(config)
        bak = self.models_bak.New_model_darcy(config)
        self._compare_forward_single_output(new, bak, x_coor, y_coor, par)

    def test_deeponet_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)

        new = self.models_new.DeepONet_plate(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_plate(config, num_input_dim=13)
        self._compare_forward_two_outputs(new, bak, x_coor, y_coor, par)

    def test_improved_deeponet_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)

        new = self.models_new.Improved_DeepONet_plate(config, num_input_dim=13)
        bak = self.models_bak.Improved_DeepONet_plate(config, num_input_dim=13)
        self._compare_forward_two_outputs(new, bak, x_coor, y_coor, par)

    def test_dcon_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)

        new = self.models_new.DCON_plate(config)
        bak = self.models_bak.DCON_plate(config)
        self._compare_forward_two_outputs(new, bak, x_coor, y_coor, par)

    def test_new_model_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)

        new = self.models_new.New_model_plate(config)
        bak = self.models_bak.New_model_plate(config)
        self._compare_forward_two_outputs(new, bak, x_coor, y_coor, par)


class TestModelsParameterEquivalence(unittest.TestCase):
    """Test that refactored models have identical parameter counts to original models."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        import sys

        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        import models as models_new  # noqa: E402

        cls.models_new = models_new
        cls.models_bak = _load_bak_module()

    def _compare_parameter_counts(
        self, new_model, bak_model, model_name: str
    ):
        """Compare parameter counts between new and old models."""
        # Get model type for state dict translation
        model_type = self._get_model_type(new_model)

        # Filter dead code from old model (e.g., DCON_plate's lift, val_lift, DCON_darcy's FC4u)
        bak_state_dict = filter_dead_code_state_dict(bak_model.state_dict())
        bak_filtered_params = sum(v.numel() for v in bak_state_dict.values())

        new_params = _count_parameters(new_model)
        bak_params = _count_parameters(bak_model)

        # For models with dead code (DCON_plate, DCON_darcy), use filtered count
        if model_type in ('dcon_plate', 'dcon_darcy'):
            expected_total = bak_filtered_params
        else:
            expected_total = bak_params["total"]

        # Check total parameters match (after filtering dead code)
        self.assertEqual(
            new_params["total"],
            expected_total,
            f"{model_name}: Total parameters mismatch - "
            f"new={new_params['total']:,}, old={expected_total:,}",
        )

        # Check trainable parameters match
        self.assertEqual(
            new_params["trainable"],
            new_params["total"],  # All params should be trainable
            f"{model_name}: Not all parameters are trainable",
        )

        # For models with same layer names (Improved_DeepONet, New_model), verify breakdown
        new_breakdown = _parameter_breakdown(new_model)
        bak_breakdown = _parameter_breakdown(bak_model)

        # Only compare layer-by-layer for models with identical structure
        if model_type is None:  # Models that don't need translation
            self.assertEqual(
                set(new_breakdown.keys()),
                set(bak_breakdown.keys()),
                f"{model_name}: Parameter layer names differ between implementations",
            )

            for layer_name in new_breakdown:
                new_count = new_breakdown[layer_name]["count"]
                bak_count = bak_breakdown[layer_name]["count"]
                self.assertEqual(
                    new_count,
                    bak_count,
                    f"{model_name}.{layer_name}: Parameter count mismatch - "
                    f"new={new_count}, old={bak_count}",
                )

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'DeepONet_darcy': 'deeponet_darcy',
            'DeepONet_plate': 'deeponet_plate',
            'Improved_DeepOnet_darcy': 'improved_deeponet_darcy',
            'Improved_DeepONet_plate': 'improved_deeponet_plate',
            'DCON_darcy': 'dcon_darcy',
            'DCON_plate': 'dcon_plate',
        }
        return mapping.get(class_name, None)

    def test_deeponet_darcy_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DeepONet_darcy(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_darcy(config, input_dim=13)
        self._compare_parameter_counts(new, bak, "DeepONet_darcy")

    def test_improved_deeponet_darcy_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.Improved_DeepOnet_darcy(config, input_dim=13)
        bak = self.models_bak.Improved_DeepOnet_darcy(config, input_dim=13)
        self._compare_parameter_counts(new, bak, "Improved_DeepOnet_darcy")

    def test_dcon_darcy_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_darcy(config)
        bak = self.models_bak.DCON_darcy(config)
        self._compare_parameter_counts(new, bak, "DCON_darcy")

    def test_new_model_darcy_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.New_model_darcy(config)
        bak = self.models_bak.New_model_darcy(config)
        self._compare_parameter_counts(new, bak, "New_model_darcy")

    def test_deeponet_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DeepONet_plate(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_plate(config, num_input_dim=13)
        self._compare_parameter_counts(new, bak, "DeepONet_plate")

    def test_improved_deeponet_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.Improved_DeepONet_plate(config, num_input_dim=13)
        bak = self.models_bak.Improved_DeepONet_plate(config, num_input_dim=13)
        self._compare_parameter_counts(new, bak, "Improved_DeepONet_plate")

    def test_dcon_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_plate(config)
        bak = self.models_bak.DCON_plate(config)
        self._compare_parameter_counts(new, bak, "DCON_plate")

    def test_new_model_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.New_model_plate(config)
        bak = self.models_bak.New_model_plate(config)
        self._compare_parameter_counts(new, bak, "New_model_plate")


class TestModelsGradientEquivalence(unittest.TestCase):
    """Test that gradients computed during backpropagation are identical."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        import models as models_new  # noqa: E402
        cls.models_new = models_new
        cls.models_bak = _load_bak_module()

    def _make_inputs(self, B=3, M=17, N=13, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        return x_coor, y_coor, par

    def _compare_gradients_single_output(self, new_model, bak_model, x_coor, y_coor, par):
        """Compare gradients for single-output models."""
        model_type = self._get_model_type(new_model)

        # Translate and load state dict
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        # Create clones for gradient computation
        new_model = copy.deepcopy(new_model)
        bak_model = copy.deepcopy(bak_model)

        # Enable gradient computation
        x_coor_new = x_coor.clone().requires_grad_(True)
        y_coor_new = y_coor.clone().requires_grad_(True)
        par_new = par.clone().requires_grad_(True)

        x_coor_bak = x_coor.clone().requires_grad_(True)
        y_coor_bak = y_coor.clone().requires_grad_(True)
        par_bak = par.clone().requires_grad_(True)

        # Forward pass
        out_new = new_model(x_coor_new, y_coor_new, par_new)
        out_bak = bak_model(x_coor_bak, y_coor_bak, par_bak)

        # Create loss and backward
        loss_new = out_new.sum()
        loss_bak = out_bak.sum()

        loss_new.backward()
        loss_bak.backward()

        # Check output gradients match
        torch.testing.assert_close(x_coor_new.grad, x_coor_bak.grad, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(y_coor_new.grad, y_coor_bak.grad, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(par_new.grad, par_bak.grad, rtol=1e-5, atol=1e-7)

        # Check parameter gradients match (for translated keys)
        for new_name, new_param in new_model.named_parameters():
            bak_name = self._translate_param_name(new_name, model_type)
            if bak_name is not None:
                bak_param = dict(bak_model.named_parameters()).get(bak_name)
                if bak_param is not None and bak_param.grad is not None:
                    self.assertIsNotNone(new_param.grad, f"New param {new_name} has no grad")
                    torch.testing.assert_close(
                        new_param.grad, bak_param.grad, rtol=1e-5, atol=1e-7,
                    )

    def _compare_gradients_two_outputs(self, new_model, bak_model, x_coor, y_coor, par):
        """Compare gradients for two-output models."""
        model_type = self._get_model_type(new_model)

        # Translate and load state dict
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        # Create clones for gradient computation
        new_model = copy.deepcopy(new_model)
        bak_model = copy.deepcopy(bak_model)

        # Enable gradient computation
        x_coor_new = x_coor.clone().requires_grad_(True)
        y_coor_new = y_coor.clone().requires_grad_(True)
        par_new = par.clone().requires_grad_(True)

        x_coor_bak = x_coor.clone().requires_grad_(True)
        y_coor_bak = y_coor.clone().requires_grad_(True)
        par_bak = par.clone().requires_grad_(True)

        # Forward pass
        out_new = new_model(x_coor_new, y_coor_new, par_new)
        out_bak = bak_model(x_coor_bak, y_coor_bak, par_bak)

        # Create loss and backward
        loss_new = out_new[0].sum() + out_new[1].sum()
        loss_bak = out_bak[0].sum() + out_bak[1].sum()

        loss_new.backward()
        loss_bak.backward()

        # Check output gradients match
        torch.testing.assert_close(x_coor_new.grad, x_coor_bak.grad, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(y_coor_new.grad, y_coor_bak.grad, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(par_new.grad, par_bak.grad, rtol=1e-5, atol=1e-7)

    def _translate_param_name(self, new_name, model_type):
        """Translate new parameter name to old format."""
        if model_type == 'deeponet_darcy':
            if new_name.startswith('branch.0.'):
                return new_name.replace('branch.0.', 'branch.')
            elif new_name.startswith('trunk.0.'):
                return new_name.replace('trunk.0.', 'trunk.')
        elif model_type == 'deeponet_plate':
            if new_name.startswith('branch.0.'):
                return new_name.replace('branch.0.', 'branch1.')
            elif new_name.startswith('branch.1.'):
                return new_name.replace('branch.1.', 'branch2.')
            elif new_name.startswith('trunk.0.'):
                return new_name.replace('trunk.0.', 'trunk1.')
            elif new_name.startswith('trunk.1.'):
                return new_name.replace('trunk.1.', 'trunk2.')
        elif model_type == 'improved_deeponet_darcy':
            if new_name.startswith('FC.0.0.'):
                return new_name.replace('FC.0.0.', 'FC1c.')
            elif new_name.startswith('FC.0.1.'):
                return new_name.replace('FC.0.1.', 'FC2c.')
            elif new_name.startswith('FC.0.2.'):
                return new_name.replace('FC.0.2.', 'FC3c.')
        elif model_type == 'improved_deeponet_plate':
            if new_name.startswith('FC.0.0.'):
                return new_name.replace('FC.0.0.', 'FC1c1.')
            elif new_name.startswith('FC.0.1.'):
                return new_name.replace('FC.0.1.', 'FC2c1.')
            elif new_name.startswith('FC.0.2.'):
                return new_name.replace('FC.0.2.', 'FC3c1.')
            elif new_name.startswith('FC.1.0.'):
                return new_name.replace('FC.1.0.', 'FC1c2.')
            elif new_name.startswith('FC.1.1.'):
                return new_name.replace('FC.1.1.', 'FC2c2.')
            elif new_name.startswith('FC.1.2.'):
                return new_name.replace('FC.1.2.', 'FC3c2.')
        elif model_type == 'dcon_darcy':
            if new_name.startswith('FC.0.0.'):
                return new_name.replace('FC.0.0.', 'FC1u.')
            elif new_name.startswith('FC.0.1.'):
                return new_name.replace('FC.0.1.', 'FC2u.')
            elif new_name.startswith('FC.0.2.'):
                return new_name.replace('FC.0.2.', 'FC3u.')
        elif model_type == 'dcon_plate':
            if new_name.startswith('FC.0.0.'):
                return new_name.replace('FC.0.0.', 'FC1u.')
            elif new_name.startswith('FC.0.1.'):
                return new_name.replace('FC.0.1.', 'FC2u.')
            elif new_name.startswith('FC.0.2.'):
                return new_name.replace('FC.0.2.', 'FC3u.')
            elif new_name.startswith('FC.1.0.'):
                return new_name.replace('FC.1.0.', 'FC1v.')
            elif new_name.startswith('FC.1.1.'):
                return new_name.replace('FC.1.1.', 'FC2v.')
            elif new_name.startswith('FC.1.2.'):
                return new_name.replace('FC.1.2.', 'FC3v.')
        return new_name  # No translation needed

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'DeepONet_darcy': 'deeponet_darcy',
            'DeepONet_plate': 'deeponet_plate',
            'Improved_DeepOnet_darcy': 'improved_deeponet_darcy',
            'Improved_DeepONet_plate': 'improved_deeponet_plate',
            'DCON_darcy': 'dcon_darcy',
            'DCON_plate': 'dcon_plate',
        }
        return mapping.get(class_name, None)

    def test_deeponet_darcy_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.DeepONet_darcy(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_darcy(config, input_dim=13)
        self._compare_gradients_single_output(new, bak, x_coor, y_coor, par)

    def test_improved_deeponet_darcy_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.Improved_DeepOnet_darcy(config, input_dim=13)
        bak = self.models_bak.Improved_DeepOnet_darcy(config, input_dim=13)
        self._compare_gradients_single_output(new, bak, x_coor, y_coor, par)

    def test_dcon_darcy_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.DCON_darcy(config)
        bak = self.models_bak.DCON_darcy(config)
        self._compare_gradients_single_output(new, bak, x_coor, y_coor, par)

    def test_new_model_darcy_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.New_model_darcy(config)
        bak = self.models_bak.New_model_darcy(config)
        self._compare_gradients_single_output(new, bak, x_coor, y_coor, par)

    def test_deeponet_plate_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.DeepONet_plate(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_plate(config, num_input_dim=13)
        self._compare_gradients_two_outputs(new, bak, x_coor, y_coor, par)

    def test_improved_deeponet_plate_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.Improved_DeepONet_plate(config, num_input_dim=13)
        bak = self.models_bak.Improved_DeepONet_plate(config, num_input_dim=13)
        self._compare_gradients_two_outputs(new, bak, x_coor, y_coor, par)

    def test_dcon_plate_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.DCON_plate(config)
        bak = self.models_bak.DCON_plate(config)
        self._compare_gradients_two_outputs(new, bak, x_coor, y_coor, par)

    def test_new_model_plate_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.New_model_plate(config)
        bak = self.models_bak.New_model_plate(config)
        self._compare_gradients_two_outputs(new, bak, x_coor, y_coor, par)


class TestModelsMultiConfiguration(unittest.TestCase):
    """Test equivalence across multiple random configurations and input sizes."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        import models as models_new  # noqa: E402
        cls.models_new = models_new
        cls.models_bak = _load_bak_module()

    def _make_inputs(self, B=3, M=17, N=13, device="cpu", dtype=torch.float32, seed=1234):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        return x_coor, y_coor, par

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'DeepONet_darcy': 'deeponet_darcy',
            'DeepONet_plate': 'deeponet_plate',
            'Improved_DeepOnet_darcy': 'improved_deeponet_darcy',
            'Improved_DeepONet_plate': 'improved_deeponet_plate',
            'DCON_darcy': 'dcon_darcy',
            'DCON_plate': 'dcon_plate',
        }
        return mapping.get(class_name, None)

    def _test_model_multiple_configs(self, new_model_class, bak_model_class,
                                     model_args_new, model_args_bak, is_two_output=False, fix_n=False):
        """Test a model across multiple configurations."""
        # For DeepONet models, N (par's second dim) must equal input_dim
        # For DCON models, N can vary
        if fix_n:
            configs = [
                {"B": 3, "M": 17, "N": 13, "seed": 42},   # Different seed
                {"B": 2, "M": 25, "N": 13, "seed": 123},  # Medium
                {"B": 1, "M": 10, "N": 13, "seed": 999},  # Smaller
            ]
        else:
            configs = [
                {"B": 3, "M": 17, "N": 13, "seed": 42},   # Different seed, same size
                {"B": 2, "M": 25, "N": 19, "seed": 123},  # Medium, seed 123
                {"B": 1, "M": 10, "N": 7, "seed": 999},   # Smaller, seed 999
            ]

        for cfg in configs:
            with self.subTest(**cfg):
                x_coor, y_coor, par = self._make_inputs(**cfg)

                new = new_model_class(**model_args_new)
                bak = bak_model_class(**model_args_bak)

                # Synchronize state dicts
                model_type = self._get_model_type(new)
                new_state = new.state_dict()
                translated_state = translate_state_dict_for_model(new_state, model_type)
                bak.load_state_dict(translated_state, strict=False)

                new.eval()
                bak.eval()

                with torch.no_grad():
                    out_new = new(x_coor, y_coor, par)
                    out_bak = bak(x_coor, y_coor, par)

                if is_two_output:
                    torch.testing.assert_close(out_new[0], out_bak[0], rtol=0.0, atol=0.0)
                    torch.testing.assert_close(out_new[1], out_bak[1], rtol=0.0, atol=0.0)
                else:
                    torch.testing.assert_close(out_new, out_bak, rtol=0.0, atol=0.0)

    def test_deeponet_darcy_multiple_configs(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        self._test_model_multiple_configs(
            lambda: self.models_new.DeepONet_darcy(config, num_input_dim=13),
            lambda: self.models_bak.DeepONet_darcy(config, input_dim=13),
            {}, {}, is_two_output=False, fix_n=True
        )

    def test_improved_deeponet_darcy_multiple_configs(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        self._test_model_multiple_configs(
            lambda: self.models_new.Improved_DeepOnet_darcy(config, input_dim=13),
            lambda: self.models_bak.Improved_DeepOnet_darcy(config, input_dim=13),
            {}, {}, is_two_output=False, fix_n=True
        )

    def test_dcon_darcy_multiple_configs(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        self._test_model_multiple_configs(
            lambda: self.models_new.DCON_darcy(config),
            lambda: self.models_bak.DCON_darcy(config),
            {}, {}, is_two_output=False
        )

    def test_dcon_plate_multiple_configs(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        self._test_model_multiple_configs(
            lambda: self.models_new.DCON_plate(config),
            lambda: self.models_bak.DCON_plate(config),
            {}, {}, is_two_output=True
        )


class TestModelsTrainingStepEquivalence(unittest.TestCase):
    """Test that a single training step produces identical parameter updates."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        import models as models_new  # noqa: E402
        cls.models_new = models_new
        cls.models_bak = _load_bak_module()

    def _make_inputs(self, B=3, M=17, N=13, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        return x_coor, y_coor, par

    def _compare_training_step(self, new_model, bak_model, x_coor, y_coor, par, lr=0.001):
        """Compare parameter updates after one training step."""
        model_type = self._get_model_type(new_model)

        # Translate and load state dict
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        # Create optimizers with same settings
        opt_new = torch.optim.SGD(new_model.parameters(), lr=lr)
        opt_bak = torch.optim.SGD(bak_model.parameters(), lr=lr)

        # Training step
        new_model.train()
        bak_model.train()

        out_new = new_model(x_coor, y_coor, par)
        out_bak = bak_model(x_coor, y_coor, par)

        loss_new = out_new.mean()
        loss_bak = out_bak.mean()

        opt_new.zero_grad()
        opt_bak.zero_grad()

        loss_new.backward()
        loss_bak.backward()

        opt_new.step()
        opt_bak.step()

        # Check parameters after update match
        new_state_after = new_model.state_dict()
        bak_state_after = filter_dead_code_state_dict(bak_model.state_dict())
        translated_after = translate_state_dict_for_model(new_state_after, model_type)

        for key in translated_after:
            if key in bak_state_after:
                torch.testing.assert_close(
                    translated_after[key], bak_state_after[key],
                    rtol=1e-5, atol=1e-7,
                )

    def _compare_training_step_two_outputs(self, new_model, bak_model, x_coor, y_coor, par, lr=0.001):
        """Compare parameter updates after one training step for two-output models."""
        model_type = self._get_model_type(new_model)

        # Translate and load state dict
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        # Create optimizers with same settings
        opt_new = torch.optim.SGD(new_model.parameters(), lr=lr)
        opt_bak = torch.optim.SGD(bak_model.parameters(), lr=lr)

        # Training step
        new_model.train()
        bak_model.train()

        out_new = new_model(x_coor, y_coor, par)
        out_bak = bak_model(x_coor, y_coor, par)

        loss_new = (out_new[0].mean() + out_new[1].mean()) / 2
        loss_bak = (out_bak[0].mean() + out_bak[1].mean()) / 2

        opt_new.zero_grad()
        opt_bak.zero_grad()

        loss_new.backward()
        loss_bak.backward()

        opt_new.step()
        opt_bak.step()

        # Check parameters after update match
        new_state_after = new_model.state_dict()
        bak_state_after = filter_dead_code_state_dict(bak_model.state_dict())
        translated_after = translate_state_dict_for_model(new_state_after, model_type)

        for key in translated_after:
            if key in bak_state_after:
                torch.testing.assert_close(
                    translated_after[key], bak_state_after[key],
                    rtol=1e-5, atol=1e-7,
                )

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'DeepONet_darcy': 'deeponet_darcy',
            'DeepONet_plate': 'deeponet_plate',
            'Improved_DeepOnet_darcy': 'improved_deeponet_darcy',
            'Improved_DeepONet_plate': 'improved_deeponet_plate',
            'DCON_darcy': 'dcon_darcy',
            'DCON_plate': 'dcon_plate',
        }
        return mapping.get(class_name, None)

    def test_deeponet_darcy_training_step(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.DeepONet_darcy(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_darcy(config, input_dim=13)
        self._compare_training_step(new, bak, x_coor, y_coor, par)

    def test_dcon_darcy_training_step(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.DCON_darcy(config)
        bak = self.models_bak.DCON_darcy(config)
        self._compare_training_step(new, bak, x_coor, y_coor, par)

    def test_dcon_plate_training_step(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par = self._make_inputs(N=13)
        new = self.models_new.DCON_plate(config)
        bak = self.models_bak.DCON_plate(config)
        self._compare_training_step_two_outputs(new, bak, x_coor, y_coor, par)


class TestModelsNumericalStability(unittest.TestCase):
    """Test equivalence with edge cases and numerical precision variations."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        import models as models_new  # noqa: E402
        cls.models_new = models_new
        cls.models_bak = _load_bak_module()

    def _make_inputs(self, B=3, M=17, N=13, device="cpu", dtype=torch.float32, seed=1234):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        return x_coor, y_coor, par

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'DeepONet_darcy': 'deeponet_darcy',
            'DeepONet_plate': 'deeponet_plate',
            'Improved_DeepOnet_darcy': 'improved_deeponet_darcy',
            'Improved_DeepONet_plate': 'improved_deeponet_plate',
            'DCON_darcy': 'dcon_darcy',
            'DCON_plate': 'dcon_plate',
        }
        return mapping.get(class_name, None)

    def _test_with_extreme_values(self, new_model, bak_model):
        """Test with extreme input values."""
        # Synchronize state dicts first
        model_type = self._get_model_type(new_model)
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        # Test with very small values
        x_coor = torch.ones(2, 5) * 1e-6
        y_coor = torch.ones(2, 5) * 1e-6
        # DeepONet models need N=input_dim=13, DCON can use any N
        n_val = 13 if 'deeponet' in model_type.lower() or 'improved' in model_type.lower() else 10
        par = torch.ones(2, n_val, 3) * 1e-6

        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new = new_model(x_coor, y_coor, par)
            out_bak = bak_model(x_coor, y_coor, par)

        torch.testing.assert_close(out_new, out_bak, rtol=1e-4, atol=1e-6)

        # Test with very large values
        x_coor = torch.ones(2, 5) * 10
        y_coor = torch.ones(2, 5) * 10
        par = torch.ones(2, n_val, 3) * 10

        with torch.no_grad():
            out_new = new_model(x_coor, y_coor, par)
            out_bak = bak_model(x_coor, y_coor, par)

        torch.testing.assert_close(out_new, out_bak, rtol=1e-4, atol=1e-6)

    def _test_with_mixed_values(self, new_model, bak_model):
        """Test with mixed positive/negative values."""
        # Synchronize state dicts first
        model_type = self._get_model_type(new_model)
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        # DeepONet models need N=input_dim=13, DCON can use any N
        n_val = 13 if 'deeponet' in model_type.lower() or 'improved' in model_type.lower() else 10
        g = torch.Generator(device='cpu')
        g.manual_seed(42)
        x_coor = torch.randn(2, 5, generator=g) * 5  # Larger variance
        y_coor = torch.randn(2, 5, generator=g) * 5
        par = torch.randn(2, n_val, 3, generator=g) * 5

        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new = new_model(x_coor, y_coor, par)
            out_bak = bak_model(x_coor, y_coor, par)

        torch.testing.assert_close(out_new, out_bak, rtol=1e-5, atol=1e-7)

    def test_deeponet_darcy_numerical_stability(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DeepONet_darcy(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_darcy(config, input_dim=13)
        self._test_with_extreme_values(new, bak)
        self._test_with_mixed_values(new, bak)

    def test_dcon_darcy_numerical_stability(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_darcy(config)
        bak = self.models_bak.DCON_darcy(config)
        self._test_with_extreme_values(new, bak)
        self._test_with_mixed_values(new, bak)

    def test_dcon_plate_numerical_stability(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_plate(config)
        bak = self.models_bak.DCON_plate(config)

        # Synchronize state dicts first
        model_type = self._get_model_type(new)
        new_state = new.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak.load_state_dict(translated_state, strict=False)

        # Test with extreme values for two-output model
        x_coor = torch.ones(2, 5) * 1e-6
        y_coor = torch.ones(2, 5) * 1e-6
        par = torch.ones(2, 10, 3) * 1e-6

        new.eval()
        bak.eval()

        with torch.no_grad():
            out_new = new(x_coor, y_coor, par)
            out_bak = bak(x_coor, y_coor, par)

        torch.testing.assert_close(out_new[0], out_bak[0], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(out_new[1], out_bak[1], rtol=1e-4, atol=1e-6)


class TestModelsCheckpointCompatibility(unittest.TestCase):
    """Test that new models can load old checkpoints and vice versa."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        import models as models_new  # noqa: E402
        cls.models_new = models_new
        cls.models_bak = _load_bak_module()

        # Create temp directory for checkpoints
        import tempfile
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _make_inputs(self, B=3, M=17, N=13, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        return x_coor, y_coor, par

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'DeepONet_darcy': 'deeponet_darcy',
            'DeepONet_plate': 'deeponet_plate',
            'Improved_DeepOnet_darcy': 'improved_deeponet_darcy',
            'Improved_DeepONet_plate': 'improved_deeponet_plate',
            'DCON_darcy': 'dcon_darcy',
            'DCON_plate': 'dcon_plate',
        }
        return mapping.get(class_name, None)

    def _test_checkpoint_roundtrip(self, new_model, bak_model, model_type):
        """Test saving old model checkpoint and loading into new model."""
        x_coor, y_coor, par = self._make_inputs(N=13)

        # Synchronize weights: translate new model state to old format
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        # Get reference output from synchronized old model
        bak_model.eval()
        with torch.no_grad():
            ref_out = bak_model(x_coor, y_coor, par)

        # Save old model checkpoint (with synchronized weights)
        old_checkpoint_path = os.path.join(self.temp_dir, 'old_model.pt')
        torch.save(bak_model.state_dict(), old_checkpoint_path)

        # Create a fresh new model instance
        if model_type == 'deeponet_darcy':
            new_model2 = self.models_new.DeepONet_darcy(new_model.config, num_input_dim=13)
        elif model_type == 'deeponet_plate':
            new_model2 = self.models_new.DeepONet_plate(new_model.config, num_input_dim=13)
        elif model_type == 'improved_deeponet_darcy':
            new_model2 = self.models_new.Improved_DeepOnet_darcy(new_model.config, input_dim=13)
        elif model_type == 'improved_deeponet_plate':
            new_model2 = self.models_new.Improved_DeepONet_plate(new_model.config, num_input_dim=13)
        elif model_type == 'dcon_darcy':
            new_model2 = self.models_new.DCON_darcy(new_model.config)
        elif model_type == 'dcon_plate':
            new_model2 = self.models_new.DCON_plate(new_model.config)
        else:
            self.skipTest(f"Unknown model type: {model_type}")

        # Load the old checkpoint and translate to new format
        old_state = torch.load(old_checkpoint_path)
        old_state = filter_dead_code_state_dict(old_state)
        translated_old_to_new = reverse_translate_state_dict_for_model(old_state, model_type)

        # Load translated checkpoint into the fresh new model
        new_model2.load_state_dict(translated_old_to_new, strict=False)

        new_model2.eval()
        with torch.no_grad():
            new_out = new_model2(x_coor, y_coor, par)

        # Outputs should match since we loaded the same weights
        if isinstance(ref_out, tuple):
            torch.testing.assert_close(new_out[0], ref_out[0], rtol=0.0, atol=0.0)
            torch.testing.assert_close(new_out[1], ref_out[1], rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(new_out, ref_out, rtol=0.0, atol=0.0)

    def test_deeponet_darcy_checkpoint_compatibility(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DeepONet_darcy(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_darcy(config, input_dim=13)
        self._test_checkpoint_roundtrip(new, bak, 'deeponet_darcy')

    def test_dcon_darcy_checkpoint_compatibility(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_darcy(config)
        bak = self.models_bak.DCON_darcy(config)
        self._test_checkpoint_roundtrip(new, bak, 'dcon_darcy')

    def test_dcon_plate_checkpoint_compatibility(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_plate(config)
        bak = self.models_bak.DCON_plate(config)
        self._test_checkpoint_roundtrip(new, bak, 'dcon_plate')


class TestModelsEvalModeConsistency(unittest.TestCase):
    """Test consistency of eval mode, dropout (if any), and multiple forward passes."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        import models as models_new  # noqa: E402
        cls.models_new = models_new
        cls.models_bak = _load_bak_module()

    def _make_inputs(self, B=3, M=17, N=13, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        return x_coor, y_coor, par

    def _test_multiple_forward_passes(self, new_model, bak_model):
        """Test that multiple forward passes in eval mode produce identical results."""
        model_type = self._get_model_type(new_model)

        # Translate and load state dict
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        x_coor, y_coor, par = self._make_inputs(N=13)

        new_model.eval()
        bak_model.eval()

        # Run multiple forward passes
        with torch.no_grad():
            for _ in range(5):
                out_new = new_model(x_coor, y_coor, par)
                out_bak = bak_model(x_coor, y_coor, par)

                if isinstance(out_new, tuple):
                    torch.testing.assert_close(out_new[0], out_bak[0], rtol=0.0, atol=0.0)
                    torch.testing.assert_close(out_new[1], out_bak[1], rtol=0.0, atol=0.0)
                else:
                    torch.testing.assert_close(out_new, out_bak, rtol=0.0, atol=0.0)

    def _test_train_eval_difference(self, new_model, bak_model):
        """Test that train and eval modes behave consistently."""
        model_type = self._get_model_type(new_model)

        # Translate and load state dict
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        x_coor, y_coor, par = self._make_inputs(N=13)

        # Test eval mode
        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new_eval = new_model(x_coor, y_coor, par)
            out_bak_eval = bak_model(x_coor, y_coor, par)

        if isinstance(out_new_eval, tuple):
            torch.testing.assert_close(out_new_eval[0], out_bak_eval[0], rtol=0.0, atol=0.0)
            torch.testing.assert_close(out_new_eval[1], out_bak_eval[1], rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(out_new_eval, out_bak_eval, rtol=0.0, atol=0.0)

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'DeepONet_darcy': 'deeponet_darcy',
            'DeepONet_plate': 'deeponet_plate',
            'Improved_DeepOnet_darcy': 'improved_deeponet_darcy',
            'Improved_DeepONet_plate': 'improved_deeponet_plate',
            'DCON_darcy': 'dcon_darcy',
            'DCON_plate': 'dcon_plate',
        }
        return mapping.get(class_name, None)

    def test_deeponet_darcy_eval_consistency(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DeepONet_darcy(config, num_input_dim=13)
        bak = self.models_bak.DeepONet_darcy(config, input_dim=13)
        self._test_multiple_forward_passes(new, bak)
        self._test_train_eval_difference(new, bak)

    def test_dcon_darcy_eval_consistency(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_darcy(config)
        bak = self.models_bak.DCON_darcy(config)
        self._test_multiple_forward_passes(new, bak)
        self._test_train_eval_difference(new, bak)

    def test_dcon_plate_eval_consistency(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_plate(config)
        bak = self.models_bak.DCON_plate(config)
        self._test_multiple_forward_passes(new, bak)
        self._test_train_eval_difference(new, bak)


class TestModelsStateDictRoundtrip(unittest.TestCase):
    """Test that state dicts can be translated back and forth without loss."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        import models as models_new  # noqa: E402
        cls.models_new = models_new
        cls.models_bak = _load_bak_module()

    def _test_state_dict_roundtrip(self, new_model, model_type):
        """Test that translating state dict to old format and back preserves values."""
        original_state = new_model.state_dict()

        # Translate new -> old -> new and verify nothing changes
        old_format = translate_state_dict_for_model(original_state, model_type)
        new_format = reverse_translate_state_dict_for_model(old_format, model_type)

        # Create a new model instance
        if model_type == 'deeponet_darcy':
            new_model2 = self.models_new.DeepONet_darcy(new_model.config, num_input_dim=13)
        elif model_type == 'deeponet_plate':
            new_model2 = self.models_new.DeepONet_plate(new_model.config, num_input_dim=13)
        elif model_type == 'improved_deeponet_darcy':
            new_model2 = self.models_new.Improved_DeepOnet_darcy(new_model.config, input_dim=13)
        elif model_type == 'improved_deeponet_plate':
            new_model2 = self.models_new.Improved_DeepONet_plate(new_model.config, num_input_dim=13)
        elif model_type == 'dcon_darcy':
            new_model2 = self.models_new.DCON_darcy(new_model.config)
        elif model_type == 'dcon_plate':
            new_model2 = self.models_new.DCON_plate(new_model.config)
        else:
            return  # Skip unsupported models

        # Load the translated-back state dict into the fresh model
        new_model2.load_state_dict(new_format, strict=True)

        # Compare state dicts
        self.assertEqual(set(original_state.keys()), set(new_model2.state_dict().keys()))
        for key in original_state:
            torch.testing.assert_close(
                original_state[key], new_model2.state_dict()[key],
                rtol=0.0, atol=0.0
            )

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'DeepONet_darcy': 'deeponet_darcy',
            'DeepONet_plate': 'deeponet_plate',
            'Improved_DeepOnet_darcy': 'improved_deeponet_darcy',
            'Improved_DeepONet_plate': 'improved_deeponet_plate',
            'DCON_darcy': 'dcon_darcy',
            'DCON_plate': 'dcon_plate',
        }
        return mapping.get(class_name, None)

    def test_deeponet_darcy_state_dict_roundtrip(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DeepONet_darcy(config, num_input_dim=13)
        model_type = self._get_model_type(new)
        self._test_state_dict_roundtrip(new, model_type)

    def test_dcon_darcy_state_dict_roundtrip(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_darcy(config)
        model_type = self._get_model_type(new)
        self._test_state_dict_roundtrip(new, model_type)

    def test_dcon_plate_state_dict_roundtrip(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_new.DCON_plate(config)
        model_type = self._get_model_type(new)
        self._test_state_dict_roundtrip(new, model_type)


if __name__ == "__main__":
    unittest.main(verbosity=2)
