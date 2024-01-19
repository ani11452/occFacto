import torch 
import numpy as np
import mcubes
import trimesh
import multiprocessing
import os

from torch.optim import lr_scheduler
from occFacto.eval.eval_metrics import MeshEvaluator

# from occFacto.eval.generate import Generator

class Trainer():

    def __init__(self, encoder, train_folder, mesh_eval, mesh_bs, viz):
        # Params for generating 3D grid
        self.padding = 0.1
        self.box_size = 1 + self.padding
        self.resolution = 25

        self.device = 'cuda'

        # For evaluating mesh
        self.evaluator = MeshEvaluator()

        # For evaluating the mesh every ... 
        self.mesh_eval_epoch = mesh_eval
        self.mesh_bs = mesh_bs

        # Visualization
        self.viz = viz

        # Make P
        p = self.box_size * self.make_3d_grid((-0.5,)*3, (0.5,)*3, (self.resolution,)*3)
        self.p = p.unsqueeze(0)

        # Save the train folder
        self.train_folder = train_folder

    def validation(self, data, model, diffFacto, loss_f, epoch):
        model.eval()

        # Trackers
        loss_track = []
        bin_accuracy_track = []
        iou_track = []
        chamferL1_track = []
        chamferL2_track = []
        dist_accuracy_track = []

        # Counters
        viz = 0
        acc_accum = 0
        loss_accum = 0

        for iter, ex in enumerate(data):
            # Pass through diffFacto encoder to extract latents
            with torch.no_grad():

                # Get Latents from DiffFacto
                latents = diffFacto(ex, device="cuda")
                latents = torch.cat(tuple(latents), dim=1)

                # Get Accuracy Comparison for Train Data
                occPoints = ex["occs"][0].to("cuda")
                occPreds = model(latents, occPoints)
                occTruths = ex["occs"][1].to("cuda")

                acc_accum += self.get_accuracy(occTruths, occPreds)
                loss_accum += loss_f(occPreds, occTruths).item()

                if (iter + 1) % 16 == 0:

                    bin_accuracy_track.append(acc_accum / 16)
                    loss_track.append(loss_accum / 16)
                        
                    # Reset
                    acc_accum = 0
                    loss_accum = 0

                    # Calculate IOU, Chamfer Distance, Normal .
                    if (epoch + 1) % self.mesh_eval_epoch == 0:
                        outputs = self.eval_metrics(model, latents, ex)
                        for out in outputs:
                            if 'iou' in out:
                                iou_track.append(out['iou'])
                            if 'chamfer-L1' in out:
                                chamferL1_track.append(out['chamfer-L1'])
                            if 'chamfer-L2' in out:
                                chamferL2_track.append(out['chamfer-L2'])
                            if 'accuracy' in out:
                                dist_accuracy_track.append(out['accuracy'])

                    if (epoch + 1) % self.viz == 0 and viz < 5:
                        self.visualize(latents, ex['token'], model, epoch)
                        viz += 1

        # Val Metrics
        val_metrics = {
            "Avg_Loss": np.mean(loss_track),
            "Avg_Bin_Accuracy": np.mean(bin_accuracy_track),
            "Avg_IOU": np.mean(iou_track),
            "Avg_ChamferL1": np.mean(chamferL1_track),
            "Avg_ChamferL2": np.mean(chamferL2_track),
            "Avg_Dist_Accuracy": np.mean(dist_accuracy_track)
        }

        return val_metrics

    def get_accuracy(self, truth, preds):
        out = torch.sigmoid(preds)
        bin = (out >= 0.5).int()
        correct_predictions = torch.sum(bin == truth)
        total_predictions = bin.numel()
        accuracy = correct_predictions.item() / total_predictions
        return accuracy

    def eval_points(self, model, latents, one):
        '''
        Evaluates occupancy values for the points

        p (tensor): points (100, 100, 100)
        z (tensor): latent code (256, 1)
        '''
        model.eval()

        # Random Sample of Latent for Evaluation
        if one:
            lat_idxs = np.random.randint(latents.shape[0], size=1)
        else:
            lat_idxs = np.random.randint(latents.shape[0], size=self.mesh_bs)

        # p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        p = self.p.to("cuda")
        with torch.no_grad():
            for i in lat_idxs:
                latent = latents[i].unsqueeze(0)

                occ_h = model(latent, p)
                occ_hats.append((occ_h.detach().cpu(), i))

        return occ_hats


    def generate_mesh(self, model, latents, viz=False):
        occ_hats = self.eval_points(model, latents, one=viz)

        all_verts = []
        all_triangles = []
        idxs = []

        for occ_hat, i in occ_hats:
            # Processes the points to be readable by marching cubes
            occ_hat = occ_hat.view(self.resolution, self.resolution, self.resolution).numpy()

            occ_hat = mcubes.smooth(occ_hat)

            # Apply marching cubes algorithm
            vertices, triangles = mcubes.marching_cubes(occ_hat, 0)

            # Normalization
            n_x, n_y, n_z = occ_hat.shape
            vertices -= 1.5
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = self.box_size * (vertices - 0.5)

            # Add
            all_verts.append(vertices)
            all_triangles.append(triangles)
            idxs.append(i)

        # TODO: change the path name here
        # mcubes.export_mesh(vertices, triangles, "occFactoDiffFreezeTraining2/best_model_pred.dae")

        return all_verts, all_triangles, idxs
    
    def save_mesh(self, vertices, triangles, path):
        mcubes.export_mesh(vertices, triangles, path)

    # Function to evaluate the mesh
    def eval_metrics(self, model, latents, ex):
        all_vertices, all_triangles, idxs = self.generate_mesh(model, latents, viz=True)
        vsaandts = zip(all_vertices, all_triangles)

        res = []

        i = 0
        for vertices, triangles in vsaandts:
            mesh = trimesh.Trimesh(vertices, triangles, process=False)

            # # Normalize
            # min_bound = mesh.bounds[0]
            # max_bound = mesh.bounds[1]
            # dimensions = max_bound - min_bound
            # scale_factors = 1 / dimensions
            # mesh.apply_scale(scale_factors)

            idx = idxs[i]

            pointcloud_tgt = ex['pointcloud_chamfer'][idx].squeeze(0).numpy()
            normals_tgt = ex['pointcloud_chamfernorms'][idx].squeeze(0).numpy()
            points_tgt = ex['occs'][0][idx].squeeze(0).numpy()
            occ_tgt = ex['occs'][1][idx].squeeze(0).numpy()

            eval_dict_mesh = self.evaluator.eval_mesh(mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)
            res.append(eval_dict_mesh)

            i += 1

        return res


    # UTILS
    def make_3d_grid(self, bb_min, bb_max, shape):
        ''' Makes a 3D grid.

        Args:
            bb_min (tuple): bounding box minimum
            bb_max (tuple): bounding box maximum
            shape (tuple): output shape
        '''
        size = shape[0] * shape[1] * shape[2]

        pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
        pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
        pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

        pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
        pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
        pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
        p = torch.stack([pxs, pys, pzs], dim=1)

        return p

    # Estimate normals if needed (probably don't need it)
    def estimate_normals(self, vertices, occ_hat):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.resolution ** 3)

        normals = []
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals
    

    def visualize(self, latents, token, model, epoch):
        all_vertices, all_triangles, idxs = self.generate_mesh(model, latents, viz=True)

        vsaandts = zip(all_vertices, all_triangles)

        i = 0
        for vertices, triangles in vsaandts:
            idx = idxs[i]
            tok = token[idx]
            flname = tok + '_' + "epoch_" + str(epoch) + '.dae'
            path = os.path.join(self.train_folder, flname)
            self.save_mesh(vertices, triangles, path)


    def backup(self):
        pass

    def update_best_model(self):
        pass

    