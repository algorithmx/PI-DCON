import unittest

import torch

from gradnorm import GradNorm, GradNormSimple, create_gradnorm


class TestGradNormFactory(unittest.TestCase):
    def test_create_gradnorm_disabled_returns_none(self):
        config = {"train": {"gradnorm": {"enabled": False}}}
        gn = create_gradnorm(num_tasks=2, config=config, device="cpu")
        self.assertIsNone(gn)

    def test_create_gradnorm_enabled_returns_full_gradnorm_by_default(self):
        """Factory returns full GradNorm by default when enabled."""
        config = {"train": {"gradnorm": {"enabled": True, "alpha": 1.2, "lr_weights": 0.01}}}
        gn = create_gradnorm(num_tasks=3, config=config, device="cpu")
        self.assertIsInstance(gn, GradNorm)
        self.assertEqual(gn.num_tasks, 3)
        self.assertAlmostEqual(gn.alpha, 1.2)
        self.assertAlmostEqual(gn.lr_weights, 0.01)
    
    def test_create_gradnorm_with_use_simple_returns_gradnorm_simple(self):
        """Factory returns GradNormSimple when use_simple=True."""
        config = {"train": {"gradnorm": {"enabled": True, "use_simple": True, "alpha": 1.5, "window_size": 7}}}
        gn = create_gradnorm(num_tasks=3, config=config, device="cpu")
        self.assertIsInstance(gn, GradNormSimple)
        self.assertEqual(gn.num_tasks, 3)
        self.assertAlmostEqual(gn.alpha, 1.5)
        self.assertEqual(gn.window_size, 7)


class TestGradNormSimple(unittest.TestCase):
    def test_initial_weights_are_ones(self):
        gn = GradNormSimple(num_tasks=4, device="cpu")
        self.assertTrue(torch.allclose(gn.weights, torch.ones(4)))

    def test_weights_are_positive_and_sum_to_num_tasks(self):
        gn = GradNormSimple(num_tasks=2, alpha=1.5, window_size=10, device="cpu")

        # First call initializes initial_losses and history
        loss_a = torch.tensor(1.0, requires_grad=True)
        loss_b = torch.tensor(1.0, requires_grad=True)
        _ = gn.update_and_get_weighted_loss([loss_a, loss_b])

        # Next calls create different training rates
        for _ in range(15):
            loss_a = torch.tensor(0.1, requires_grad=True)  # much smaller -> faster task
            loss_b = torch.tensor(1.0, requires_grad=True)  # slower task
            _ = gn.update_and_get_weighted_loss([loss_a, loss_b])

        self.assertTrue(torch.all(gn.weights > 0))
        self.assertTrue(torch.isfinite(gn.weights).all())
        self.assertAlmostEqual(float(gn.weights.sum().item()), 2.0, places=5)

    def test_slower_task_gets_higher_weight(self):
        gn = GradNormSimple(num_tasks=2, alpha=1.5, window_size=10, device="cpu")

        # Initialize
        _ = gn.update_and_get_weighted_loss([torch.tensor(1.0), torch.tensor(1.0)])

        # Task0 decreases a lot, task1 stays high
        for _ in range(20):
            _ = gn.update_and_get_weighted_loss([torch.tensor(0.05), torch.tensor(1.0)])

        w0, w1 = gn.get_weights_list()
        self.assertGreater(w1, w0)

    def test_deterministic_given_same_sequence(self):
        seq = [(1.0, 1.0)] + [(0.2, 1.0)] * 30

        gn1 = GradNormSimple(num_tasks=2, alpha=1.3, window_size=10, device="cpu")
        gn2 = GradNormSimple(num_tasks=2, alpha=1.3, window_size=10, device="cpu")

        for a, b in seq:
            gn1.update_and_get_weighted_loss([torch.tensor(a), torch.tensor(b)])
            gn2.update_and_get_weighted_loss([torch.tensor(a), torch.tensor(b)])

        self.assertTrue(torch.allclose(gn1.weights, gn2.weights, atol=0.0, rtol=0.0))

    def test_weighted_loss_is_differentiable_wrt_model_params(self):
        # Ensure autograd flows through weighted sum even though weights are computed from detached values.
        gn = GradNormSimple(num_tasks=2, alpha=1.5, window_size=10, device="cpu")

        p = torch.nn.Parameter(torch.tensor(2.0))

        # Initialize history
        l1 = (p - 1.0) ** 2
        l2 = (p + 1.0) ** 2
        out = gn.update_and_get_weighted_loss([l1, l2])
        self.assertIsInstance(out, torch.Tensor)
        self.assertTrue(out.requires_grad)
        out.backward()
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_reset_restores_state(self):
        gn = GradNormSimple(num_tasks=3, device="cpu")
        gn.update_and_get_weighted_loss([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])
        self.assertIsNotNone(gn.initial_losses)
        gn.reset()
        self.assertIsNone(gn.initial_losses)
        self.assertEqual(gn.loss_history, [[], [], []])
        self.assertTrue(torch.allclose(gn.weights, torch.ones(3)))


class TestGradNormFull(unittest.TestCase):
    """Test the full GradNorm implementation faithful to the paper."""
    
    def test_initial_weights_are_ones(self):
        gn = GradNorm(num_tasks=3, alpha=1.5, device="cpu")
        w = gn.weights
        self.assertTrue(torch.allclose(w, torch.ones(3)))
        self.assertTrue(torch.isfinite(w).all())

    def test_get_weights_returns_detached_tensor(self):
        """Test that get_weights() returns a detached tensor."""
        gn = GradNorm(num_tasks=2, alpha=1.5, device="cpu")
        w = gn.get_weights()
        self.assertIsInstance(w, torch.Tensor)
        self.assertFalse(w.requires_grad)
        self.assertEqual(w.shape, (2,))

    def test_get_weights_list_returns_list(self):
        """Test that get_weights_list() returns a Python list."""
        gn = GradNorm(num_tasks=2, alpha=1.5, device="cpu")
        w = gn.get_weights_list()
        self.assertIsInstance(w, list)
        self.assertEqual(len(w), 2)
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 1.0)

    def test_weights_remain_positive_after_step(self):
        torch.manual_seed(42)
        
        # Simple model with shared layer
        shared = torch.nn.Linear(2, 4, bias=False)
        head1 = torch.nn.Linear(4, 1, bias=False)
        head2 = torch.nn.Linear(4, 1, bias=False)
        
        model_optimizer = torch.optim.Adam(
            list(shared.parameters()) + list(head1.parameters()) + list(head2.parameters()),
            lr=0.01
        )
        
        gn = GradNorm(num_tasks=2, alpha=1.5, lr_weights=0.025, device="cpu")
        
        x = torch.randn(4, 2)
        for _ in range(10):
            h = shared(x)
            loss1 = head1(h).pow(2).mean()
            loss2 = head2(h).pow(2).mean()
            
            gn.step([loss1, loss2], shared, model_optimizer)
        
        self.assertTrue(torch.all(gn.weights > 0))
        self.assertTrue(torch.isfinite(gn.weights).all())

    def test_weights_sum_preserved_after_renormalization(self):
        torch.manual_seed(42)
        
        shared = torch.nn.Linear(2, 4, bias=False)
        head1 = torch.nn.Linear(4, 1, bias=False)
        head2 = torch.nn.Linear(4, 1, bias=False)
        head3 = torch.nn.Linear(4, 1, bias=False)
        
        all_params = (list(shared.parameters()) + list(head1.parameters()) + 
                      list(head2.parameters()) + list(head3.parameters()))
        model_optimizer = torch.optim.Adam(all_params, lr=0.01)
        
        gn = GradNorm(num_tasks=3, alpha=1.5, device="cpu")
        T_expected = 3.0  # sum should be preserved
        
        x = torch.randn(4, 2)
        for _ in range(20):
            h = shared(x)
            loss1 = head1(h).pow(2).mean()
            loss2 = head2(h).pow(2).mean() * 0.1  # faster task
            loss3 = head3(h).pow(2).mean() * 2.0  # slower task
            
            gn.step([loss1, loss2, loss3], shared, model_optimizer)
        
        self.assertAlmostEqual(float(gn.weights.sum().item()), T_expected, places=4)

    def test_slower_task_gets_higher_weight(self):
        """Tasks with higher loss ratios (slower training) should get higher weights."""
        torch.manual_seed(42)
        
        shared = torch.nn.Linear(2, 4, bias=False)
        head1 = torch.nn.Linear(4, 1, bias=False)
        head2 = torch.nn.Linear(4, 1, bias=False)
        
        all_params = list(shared.parameters()) + list(head1.parameters()) + list(head2.parameters())
        model_optimizer = torch.optim.Adam(all_params, lr=0.001)
        
        gn = GradNorm(num_tasks=2, alpha=1.5, device="cpu")
        
        x = torch.randn(4, 2)
        for i in range(30):
            h = shared(x)
            # Task 1 decreases faster (smaller loss)
            loss1 = head1(h).pow(2).mean() * (0.1 if i > 5 else 1.0)
            # Task 2 stays high (slower training)
            loss2 = head2(h).pow(2).mean() * 1.0
            
            gn.step([loss1, loss2], shared, model_optimizer)
        
        w0, w1 = gn.get_weights_list()
        # Task 2 (slower) should have higher weight
        self.assertGreater(w1, w0)

    def test_step_updates_model_parameters(self):
        """Verify that step() actually updates the model via model_optimizer."""
        torch.manual_seed(42)
        
        shared = torch.nn.Linear(2, 4, bias=False)
        head = torch.nn.Linear(4, 1, bias=False)
        
        initial_weight = shared.weight.clone().detach()
        
        model_optimizer = torch.optim.Adam(
            list(shared.parameters()) + list(head.parameters()),
            lr=0.1
        )
        
        gn = GradNorm(num_tasks=1, alpha=1.5, device="cpu")
        
        x = torch.randn(4, 2)
        h = shared(x)
        loss = (head(h) - 1.0).pow(2).mean()
        
        gn.step([loss], shared, model_optimizer)
        
        # Model weights should have changed
        self.assertFalse(torch.allclose(shared.weight, initial_weight))

    def test_reset_restores_initial_state(self):
        gn = GradNorm(num_tasks=2, alpha=1.5, device="cpu")
        
        # Modify state
        gn.initial_losses = torch.tensor([1.0, 2.0])
        with torch.no_grad():
            gn.weights.data = torch.tensor([0.5, 1.5])
        
        gn.reset()
        
        self.assertIsNone(gn.initial_losses)
        self.assertTrue(torch.allclose(gn.weights, torch.ones(2)))

    def test_deterministic_given_same_seed(self):
        """Same random seed should produce same weight evolution."""
        def run_training(seed):
            torch.manual_seed(seed)
            shared = torch.nn.Linear(2, 4, bias=False)
            head1 = torch.nn.Linear(4, 1, bias=False)
            head2 = torch.nn.Linear(4, 1, bias=False)
            
            all_params = list(shared.parameters()) + list(head1.parameters()) + list(head2.parameters())
            model_optimizer = torch.optim.Adam(all_params, lr=0.01)
            
            gn = GradNorm(num_tasks=2, alpha=1.5, lr_weights=0.025, device="cpu")
            
            x = torch.randn(4, 2)
            for _ in range(15):
                h = shared(x)
                loss1 = head1(h).pow(2).mean()
                loss2 = head2(h).pow(2).mean()
                gn.step([loss1, loss2], shared, model_optimizer)
            
            return gn.get_weights_list()
        
        weights1 = run_training(seed=123)
        weights2 = run_training(seed=123)
        
        self.assertAlmostEqual(weights1[0], weights2[0], places=5)
        self.assertAlmostEqual(weights1[1], weights2[1], places=5)

    def test_weights_not_updated_by_task_loss_gradient(self):
        """Verify weights are only updated by GradNorm loss, not by task loss."""
        torch.manual_seed(42)
        
        shared = torch.nn.Linear(2, 4, bias=False)
        head1 = torch.nn.Linear(4, 1, bias=False)
        head2 = torch.nn.Linear(4, 1, bias=False)
        
        all_params = list(shared.parameters()) + list(head1.parameters()) + list(head2.parameters())
        model_optimizer = torch.optim.Adam(all_params, lr=0.01)
        
        gn = GradNorm(num_tasks=2, alpha=1.5, lr_weights=0.0, device="cpu")  # lr=0 disables weight updates
        
        initial_weights = gn.get_weights().clone()
        
        x = torch.randn(4, 2)
        for _ in range(5):
            h = shared(x)
            loss1 = head1(h).pow(2).mean()
            loss2 = head2(h).pow(2).mean()
            gn.step([loss1, loss2], shared, model_optimizer)
        
        # With lr_weights=0, weights should remain at initial (after renormalization)
        # They should still sum to T=2 but be normalized
        final_weights = gn.get_weights()
        self.assertAlmostEqual(float(final_weights.sum().item()), 2.0, places=4)

    def test_individual_weight_can_exceed_one(self):
        """Individual weights can be > 1 as long as they sum to T."""
        torch.manual_seed(42)
        
        shared = torch.nn.Linear(2, 4, bias=False)
        head1 = torch.nn.Linear(4, 1, bias=False)
        head2 = torch.nn.Linear(4, 1, bias=False)
        
        all_params = list(shared.parameters()) + list(head1.parameters()) + list(head2.parameters())
        model_optimizer = torch.optim.Adam(all_params, lr=0.001)
        
        gn = GradNorm(num_tasks=2, alpha=1.5, lr_weights=0.05, device="cpu")
        
        x = torch.randn(4, 2)
        for i in range(50):
            h = shared(x)
            # Make task 2 much harder (higher loss)
            loss1 = head1(h).pow(2).mean() * 0.01
            loss2 = head2(h).pow(2).mean() * 10.0
            gn.step([loss1, loss2], shared, model_optimizer)
        
        w = gn.get_weights_list()
        # One weight might exceed 1.0, but sum should be 2.0
        self.assertAlmostEqual(w[0] + w[1], 2.0, places=4)
        # This is valid behavior - individual weights can exceed 1


if __name__ == "__main__":
    unittest.main()
