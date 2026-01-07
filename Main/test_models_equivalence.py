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


if __name__ == "__main__":
    unittest.main(verbosity=2)
