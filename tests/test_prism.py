"""Regression checks for PRISM checkpoint, feature, and split behavior."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from model.getdata import smiles2graph
from model.mpn_vas_st import PRISMPriorEncoder
from model.checkpoints import load_encoder
from model.fusion import make_fusion
from finetune_pair import data_builder
from finetune_mix import build_mix_model, load_checkpoint_into_mix_model, summarize_runs


class PRISMRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.graphs = smiles2graph(['CCO', 'CC(C)(C)O', 'O'])

    def encoder(self, globals_on=False, atom_messages=True):
        graph = self.graphs[0]
        return PRISMPriorEncoder(
            graph.x.shape[1], graph.edge_attr.shape[1],
            num_global_features=8 if globals_on else 0,
            hidden_dim=16, projection_dim=8, cov_num=3,
            dropout_rate=0.3, atom_messages=atom_messages).eval()

    def save_encoder(self, path, encoder):
        torch.save({'encoder_config': encoder.encoder_config(),
                    'encoder_state_dict': encoder.state_dict()}, path)

    def test_checkpoint_roundtrip_and_global_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            for globals_on in (False, True):
                for atom_messages in (False, True):
                    with self.subTest(globals_on=globals_on, atom_messages=atom_messages):
                        enc = self.encoder(globals_on, atom_messages)
                        path = Path(tmp) / 'encoder.pth'
                        self.save_encoder(path, enc)
                        if globals_on:
                            with self.assertRaisesRegex(ValueError, 'global features'):
                                load_encoder(path, self.graphs[0], 'cpu')
                        loaded = load_encoder(path, self.graphs[0], 'cpu', globals_on).eval()
                        self.assertEqual(loaded.output_dim, 48 if globals_on else 16)
                        torch.testing.assert_close(enc(self.graphs), loaded(self.graphs))

    def test_padding_does_not_change_with_batch_companions(self):
        for atom_messages in (True, False):
            enc = self.encoder(atom_messages=atom_messages)
            # Nonzero biases exercise the dummy-message regression directly.
            with torch.no_grad():
                for name, param in enc.named_parameters():
                    if name.endswith('bias'):
                        param.fill_(0.2)
            alone = enc([self.graphs[0]])
            together = enc(self.graphs)
            torch.testing.assert_close(alone[0], together[0], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(enc([self.graphs[2]])[0], together[2], atol=1e-6, rtol=1e-5)
            together.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None))

    def test_default_encoder_ignores_global_features(self):
        enc = self.encoder()
        changed = [g.clone() for g in self.graphs]
        for graph in changed:
            graph.global_features = graph.global_features + 100
        torch.testing.assert_close(enc(self.graphs), enc(changed), rtol=0, atol=0)
        enabled = self.encoder(True)
        self.assertFalse(torch.allclose(enabled(self.graphs), enabled(changed)))

    def test_fusion_global_feature_switch(self):
        embs = torch.randn(3, 2, 16)
        for kind in ('static', 'adaptive'):
            for pair in (False, True):
                fusion = make_fusion(2, 16, pair=pair, fusion_type=kind).eval()
                first = fusion(embs, torch.zeros(3, 8))[0]
                second = fusion(embs, torch.ones(3, 8) * 100)[0]
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            enabled = make_fusion(2, 16, fusion_type=kind, use_global_features=True).eval()
            self.assertFalse(torch.allclose(enabled(embs, torch.zeros(3, 8))[0],
                                            enabled(embs, torch.ones(3, 8) * 100)[0]))

    def test_train_validation_groups_are_canonical_solutes(self):
        smiles = ['CCO', 'OCC', 'CCC', 'CCCC', 'CO', 'CO', 'CCN', 'CN']
        solutes = smiles2graph(smiles)
        solvents = smiles2graph(['O'] * len(smiles))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'split.csv'
            train, val = data_builder(solutes, solvents, np.arange(8), np.full(8, 298.15),
                                      batch_size=4, seed=42, val_ratio=0.3, split_path=str(path))
            split = pd.read_csv(path)
            self.assertEqual(len(split), 8)
            self.assertEqual(set(split['split']), {'train', 'val'})
            self.assertTrue((split.groupby('canonical_solute')['split'].nunique() == 1).all())
            self.assertEqual(len(train.dataset) + len(val.dataset), 8)

    def test_warm_start_transfers_only_fusions_and_freezes_encoders(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc_path = Path(tmp) / 'encoder.pth'
            self.save_encoder(enc_path, self.encoder())
            def build(kind='adaptive'):
                return build_mix_model(self.graphs[0], self.graphs[1], [str(enc_path)], 512,
                                       0.1, 0.1, (16, 8), 0.1, 4, torch.device('cpu'), kind)
            source, target = build(), build()
            with torch.no_grad():
                for p in source.parameters():
                    p.add_(0.25)
            before = {k: v.clone() for k, v in target.state_dict().items()}
            warm = Path(tmp) / 'pure.pth'
            torch.save({'model_state_dict': source.state_dict(),
                        'fusion_type': 'adaptive', 'use_global_features': False,
                        'encoder_configs': [self.encoder().encoder_config()]}, warm)
            load_checkpoint_into_mix_model(target, str(warm), 'cpu')
            for name, actual in target.state_dict().items():
                expected = source.state_dict()[name] if '_fusion.' in name else before[name]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            target.train()
            self.assertFalse(target.solute_encs[0].training)
            self.assertFalse(target.solvent_encs[0].training)
            self.assertTrue(target.solute_fusion.training)
            with self.assertRaisesRegex(ValueError, 'fusion_type mismatch'):
                load_checkpoint_into_mix_model(build('static'), str(warm), 'cpu')
            batch = Batch.from_data_list(self.graphs)
            pred = target(batch, batch, batch, torch.full((3, 1), .5),
                          torch.full((3, 1), .5), torch.full((3, 1), 298.15))
            pred.sum().backward()
            self.assertTrue(all(p.grad is None for p in target.solute_encs.parameters()))

    def test_summary_uses_mean_rmse_and_population_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'runs.csv')
            summarize_runs([{'split_strategy': 'random', 'test_rmse': x, 'test_pct_le1': .8}
                            for x in (1., 3.)], path)
            summary = pd.read_csv(path.replace('.csv', '_summary.csv')).iloc[0]
            self.assertEqual(summary['rmse_mean'], 2.)
            self.assertEqual(summary['rmse_std'], 1.)


if __name__ == '__main__':
    unittest.main()
