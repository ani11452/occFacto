import torch 
import numpy as np
import mcubes
import trimesh
import multiprocessing

from torch.optim import lr_scheduler
from occFacto.eval.eval_metrics import MeshEvaluator

# from occFacto.eval.generate import Generator


class Trainer():

    def __init__(self):
        # Params for generating 3D grid
        self.padding = 0.1
        self.box_size = 1 + self.padding
        self.resolution = 25

        self.device = 'cuda'

        # For evaluating mesh
        self.evaluator = MeshEvaluator()

        # For evaluating the mesh every ... 
        self.mesh_eval_epoch = 2000
        self.mesh_bs = 5

    def validation(self, data, model, diffFacto, loss_f, epoch):
        model.eval()

        # Trackers
        loss_track = []
        accuracy_track = []

        for ex in data:
            # Pass through diffFacto encoder to extract latents
            with torch.no_grad():

                # Get Latents from DiffFacto
                latents = diffFacto(ex, device="cuda")
                latents = torch.cat(tuple(latents), dim=1)

                # Loss
                loss_track.append(loss_f(occPreds, occTruths).item())

                # Get Accuracy Comparison for Train Data
                occPoints = ex["occs"][0].to("cuda")
                occPreds = model(latents, occPoints)
                occTruths = ex["occs"][1].to("cuda")
                accuracy = self.get_accuracy(occTruths, occPreds)
                accuracy_track.append(accuracy)

                # Calculate IOU, Chamfer Distance, Normal .
                if ((epoch + 1) % self.mesh_eval_epoch == 0 and epoch != 0):
                        iou, chamfer, normal = self.eval_metrics(model, latents, ex)

        # Val Metrics
        val_metrics = {
            "Avg_Loss": np.mean(loss_track),
            "Avg_Accuracy": np.mean(accuracy_track)
        }

        return val_metrics


    def log(self):
        pass

    def get_accuracy(self, truth, preds):
        out = torch.sigmoid(preds)
        bin = (out >= 0.5).int()
        correct_predictions = torch.sum(bin == truth)
        total_predictions = bin.numel()
        accuracy = correct_predictions.item() / total_predictions
        return accuracy

    def metrics(self, accuracy_p=None, iou_p=None, chamfer_p=None, normal_p=None):
        metrics = {
            "accuracy": None,
            "iou": None,
            "chamfer": None,
            "normal": None
        }

        if accuracy_p:
            metrics["accuracy"] = self.get_accuracy(accuracy_p)
        if iou_p:
            metrics["iou"] = self.get_iou(iou_p)
        if chamfer_p:
            metrics["chamfer"] = self.get_iou(chamfer_p)
        if normal_p:
            metrics["normal"] = self.get_iou(normal_p)
        
        return metrics
    

    # WILLIAM ADDED CODE:

    def eval_points(self, model, latents):
        '''
        Evaluates occupancy values for the points

        p (tensor): points (100, 100, 100)
        z (tensor): latent code (256, 1)
        '''
        model.eval()

        # Random Sample of Latent for Evaluation
        lat_idxs = np.random.randint(latents.shape[0], size=self.mesh_bs)
        latents = latents[lat_idxs]
          
        p = self.box_size * self.make_3d_grid((-0.5,)*3, (0.5,)*3, (self.resolution,)*3)
        p = p.unsqueeze(0)

        # p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        p = p.to("cuda")
        with torch.no_grad():
            for i in range(self.mesh_bs):
                latent = latents[i].unsqueze(0)

                print(latent.size(), p.size())

                occ_h = model(latent, p)
                occ_hats.append(occ_h.detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat


    # def worker(self, occ_h):
    #     # Smooth and apply marching cubes
    #     occ_h = mcubes.smooth(occ_h)
    #     vertices, triangles = mcubes.marching_cubes(occ_h, 0)
    #     return vertices, triangles

    # def generate_mesh(self, model, latents):
    #     occ_hat = self.eval_points(model, latents)
    #     occ_hat = occ_hat.view(occ_hat.size(0), self.resolution, self.resolution, self.resolution).numpy()
    #     occ_hat = np.pad(occ_hat, 1, 'constant', constant_values=-1e6)

    #     # Create chunks for multiprocessing
    #     num_processes = 4  # Number of VCPUs
    #     chunk_size = int(np.ceil(occ_hat.shape[0] / num_processes))
    #     chunks = [occ_hat[i:i + chunk_size] for i in range(0, occ_hat.shape[0], chunk_size)]

    #     # Use multiprocessing to process chunks
    #     with multiprocessing.Pool(num_processes) as pool:
    #         results = pool.map(self.worker, chunks)

    #     # Combine results from all processes
    #     all_verts, all_triangles = zip(*results)
    #     all_verts = [vert for sublist in all_verts for vert in sublist]
    #     all_triangles = [tri for sublist in all_triangles for tri in sublist]

    #     print(all_verts, all_triangles)
    #     return all_verts, all_triangles


    def generate_mesh(self, model, latents):
        occ_hat = self.eval_points(model, latents)

        print(occ_hat.shape)
        return

        # Processes the points to be readable by marching cubes
        occ_hat = occ_hat.view(occ_hat.size(0), self.resolution, self.resolution, self.resolution).numpy()
        occ_hat = np.pad(occ_hat, 1, 'constant', constant_values=-1e6)
        
        # Marching cube smooth out the occupancy hat
        all_verts = []
        all_triangles = []
        for i in range(occ_hat.shape[0]):
            occ_h = occ_hat[i]
            occ_h = mcubes.smooth(occ_h)

            # Apply marching cubes algorithm
            vertices, triangles = mcubes.marching_cubes(occ_h, 0)

            # Add
            all_verts.append(vertices)
            all_triangles.append(triangles)

        # TODO: change the path name here
        # mcubes.export_mesh(vertices, triangles, "occFactoDiffFreezeTraining2/best_model_pred.dae")

        print(all_verts, all_triangles)

        return all_verts, all_triangles

        return vertices, triangles
    
    def save_mesh(self, vertices, triangles):
        mcubes.export_mesh(vertices, triangles, "occFactoDiffFreezeTraining2/best_model_pred.dae")

    # Function to evaluate the mesh
    def eval_metrics(self, model, latents, ex):
        vertices, triangles = self.generate_mesh(model, latents)
        mesh = trimesh.Trimesh(vertices, triangles, process=False)

        pointcloud_tgt = ex['pointcloud_chamfer'].squeeze(0).numpy()
        normals_tgt = ex['pointcloud_chamfernorms'].squeeze(0).numpy()
        points_tgt = ex['occs'][0].squeeze(0).numpy()
        occ_tgt = ex['occs'][1].squeeze(0).numpy()

        eval_dict_mesh = self.evaluator.eval_mesh(mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)

        return eval_dict_mesh


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

    def save_visualization(self):
        pass

    def backup(self):
        pass

    def update_best_model(self):
        pass

    