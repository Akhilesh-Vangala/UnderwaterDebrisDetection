import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from models import build_cswin_base, build_cswin_small, build_cswin_large, ResNet50Baseline, EnsembleCSWin
from data import UnderwaterGarbageDataset, UnderwaterAugmentation, UnderwaterValidationTransform, TestTimeAugmentation
from training import ModelTrainer
from evaluation import ModelEvaluator
from knowledge_distillation import KnowledgeDistillationTrainer
from ensemble import EnsembleModel, EnsembleTrainer, AdaptiveEnsemble
from utils.onnx_converter import ONNXConverter
from utils.tensorrt_converter import TensorRTConverter
from inference import InferenceEngine, FPSBenchmark
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import set_seed, setup_logging, save_config
from collections import Counter


def calculate_class_weights(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
    return weights


def main():
    parser = argparse.ArgumentParser(description='Underwater Garbage Detection & Classification')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to dataset')
    parser.add_argument('--model', type=str, default='cswin', 
                       choices=['cswin', 'resnet50', 'both', 'ensemble', 'distill'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--convert_onnx', action='store_true', help='Convert to ONNX')
    parser.add_argument('--convert_tensorrt', action='store_true', help='Convert to TensorRT')
    parser.add_argument('--benchmark_fps', action='store_true', help='Benchmark FPS')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--use_ensemble', action='store_true', help='Use ensemble methods')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss')
    parser.add_argument('--use_label_smoothing', action='store_true', default=True, help='Use label smoothing')
    parser.add_argument('--use_lookahead', action='store_true', default=True, help='Use Lookahead optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    set_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    logger = setup_logging(output_dir / 'logs')
    
    print("Underwater Garbage Detection & Classification")
    print("=" * 70)
    print("Project affiliated with New York University Center for Data Science")
    print("=" * 70)
    
    print("\n[1/6] Data Loading...")
    train_transform = UnderwaterAugmentation(
        img_size=args.img_size,
        use_haze=True, use_color_jitter=True, use_contrast=True,
        use_clahe=True, use_noise=True, use_blur=True, use_geometric=True
    )
    val_transform = UnderwaterValidationTransform(img_size=args.img_size)
    
    train_dataset = UnderwaterGarbageDataset(
        data_dir=args.data_dir,
        transform=train_transform,
        split='train'
    )
    val_dataset = UnderwaterGarbageDataset(
        data_dir=args.data_dir,
        transform=val_transform,
        split='val'
    )
    test_dataset = UnderwaterGarbageDataset(
        data_dir=args.data_dir,
        transform=val_transform,
        split='test'
    )
    
    class_weights = calculate_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")
    
    sampler = WeightedRandomSampler(
        weights=[class_weights[label] for _, label in train_dataset],
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Number of classes: {train_dataset.num_classes}")
    
    print("\n[2/6] Model Training...")
    trainer = ModelTrainer(
        model_dir=output_dir / 'models',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    device = trainer.device
    
    results = {}
    
    if args.model in ['cswin', 'both', 'ensemble', 'distill']:
        print("\nTraining CSWin Transformer...")
        cswin_model = build_cswin_base(num_classes=8, img_size=args.img_size)
        cswin_result = trainer.train(
            cswin_model, train_loader, val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            use_mixed_precision=True,
            use_lookahead=args.use_lookahead,
            use_focal_loss=args.use_focal_loss,
            use_label_smoothing=args.use_label_smoothing,
            class_weights=class_weights,
            warmup_epochs=args.warmup_epochs,
            gradient_accumulation_steps=args.gradient_accumulation,
            model_name='cswin'
        )
        results['CSWin Transformer'] = {
            'model': cswin_result['model'],
            'history': cswin_result['history'],
            'best_val_acc': cswin_result['best_val_acc']
        }
    
    if args.model in ['resnet50', 'both', 'distill']:
        print("\nTraining ResNet-50 Baseline...")
        resnet_model = ResNet50Baseline(num_classes=8, pretrained=True)
        resnet_result = trainer.train(
            resnet_model, train_loader, val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            use_mixed_precision=True,
            use_lookahead=args.use_lookahead,
            use_focal_loss=args.use_focal_loss,
            use_label_smoothing=args.use_label_smoothing,
            class_weights=class_weights,
            warmup_epochs=args.warmup_epochs,
            gradient_accumulation_steps=args.gradient_accumulation,
            model_name='resnet50'
        )
        results['ResNet-50'] = {
            'model': resnet_result['model'],
            'history': resnet_result['history'],
            'best_val_acc': resnet_result['best_val_acc']
        }
    
    if args.model == 'distill':
        print("\nTraining with Knowledge Distillation...")
        teacher_model = results['ResNet-50']['model']
        student_model = build_cswin_small(num_classes=8, img_size=args.img_size)
        
        from knowledge_distillation import KnowledgeDistillationTrainer
        distill_trainer = KnowledgeDistillationTrainer(teacher_model, student_model, device=device)
        
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate * 0.1)
        
        for epoch in range(args.epochs // 2):
            student_model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                loss = distill_trainer.train_step(images, labels, optimizer)
                train_loss += loss
            print(f"Distillation Epoch {epoch+1}: Loss: {train_loss/len(train_loader):.4f}")
        
        results['CSWin (Distilled)'] = {
            'model': student_model,
            'history': {},
            'best_val_acc': 0.0
        }
    
    if args.model == 'ensemble' or args.use_ensemble:
        print("\nTraining Ensemble Model...")
        ensemble_models = [results['CSWin Transformer']['model']]
        if 'ResNet-50' in results:
            ensemble_models.append(results['ResNet-50']['model'])
        
        ensemble = EnsembleModel(ensemble_models, weights=[0.7, 0.3] if len(ensemble_models) > 1 else None)
        ensemble = ensemble.to(device)
        results['Ensemble'] = {'model': ensemble, 'history': {}, 'best_val_acc': 0.0}
    
    print("\n[3/6] Model Evaluation...")
    evaluator = ModelEvaluator(num_classes=8)
    
    tta_transforms = None
    if args.use_tta:
        tta = TestTimeAugmentation(img_size=args.img_size)
        tta_transforms = tta.augmentations
    
    evaluation_results = {}
    for model_name, model_info in results.items():
        print(f"\nEvaluating {model_name}...")
        eval_result = evaluator.evaluate(
            model_info['model'], test_loader, device=device,
            use_tta=args.use_tta, tta_transforms=tta_transforms
        )
        evaluation_results[model_name] = eval_result
        evaluator.print_results(eval_result)
        
        evaluator.plot_confusion_matrix(
            eval_result['confusion_matrix'],
            save_path=output_dir / 'plots' / f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        )
        evaluator.plot_class_performance(
            eval_result['per_class_metrics'],
            save_path=output_dir / 'plots' / f'{model_name.lower().replace(" ", "_")}_class_performance.png'
        )
        
        eval_result['per_class_metrics'].to_csv(
            output_dir / f'{model_name.lower().replace(" ", "_")}_per_class_metrics.csv',
            index=False
        )
    
    print("\n[4/6] Model Comparison...")
    comparison_df = evaluator.compare_models(evaluation_results)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_info = results[best_model_name]
    best_model = best_model_info['model']
    
    improvement_over_baseline = 0.0
    if 'ResNet-50' in evaluation_results:
        baseline_map = evaluation_results['ResNet-50']['mean_ap']
        best_map = evaluation_results[best_model_name]['mean_ap']
        improvement_over_baseline = ((best_map - baseline_map) / baseline_map) * 100
    
    print(f"\nBest model: {best_model_name}")
    print(f"  Accuracy: {evaluation_results[best_model_name]['accuracy']*100:.2f}%")
    print(f"  mAP: {evaluation_results[best_model_name]['mean_ap']*100:.2f}%")
    if improvement_over_baseline > 0:
        print(f"  Improvement over ResNet-50: {improvement_over_baseline:.1f}%")
    
    print("\n[5/6] Model Optimization...")
    if args.convert_onnx:
        print("\nConverting to ONNX...")
        onnx_converter = ONNXConverter(best_model, input_size=(3, args.img_size, args.img_size))
        onnx_path = onnx_converter.convert(output_dir / 'onnx' / 'model.onnx')
        
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
        onnx_converter.validate(onnx_path, dummy_input)
        
        if args.benchmark_fps:
            print("\nBenchmarking ONNX FPS...")
            onnx_engine = InferenceEngine(str(onnx_path), device=device, model_type='onnx')
            benchmark = FPSBenchmark(onnx_engine, img_size=(args.img_size, args.img_size))
            fps_results = benchmark.benchmark(num_iterations=1000)
            benchmark.print_results(fps_results)
    
    if args.convert_tensorrt:
        print("\nConverting to TensorRT...")
        if not args.convert_onnx:
            onnx_path = output_dir / 'onnx' / 'model.onnx'
            onnx_converter = ONNXConverter(best_model, input_size=(3, args.img_size, args.img_size))
            onnx_path = onnx_converter.convert(onnx_path)
        
        tensorrt_converter = TensorRTConverter(str(onnx_path), input_size=(3, args.img_size, args.img_size))
        tensorrt_path = tensorrt_converter.convert(
            output_dir / 'tensorrt' / 'model.trt',
            precision='fp16'
        )
        
        if tensorrt_path and args.benchmark_fps:
            print("\nBenchmarking TensorRT FPS...")
            tensorrt_engine = InferenceEngine(str(tensorrt_path), device=device, model_type='tensorrt')
            benchmark = FPSBenchmark(tensorrt_engine, img_size=(args.img_size, args.img_size))
            fps_results = benchmark.benchmark(num_iterations=1000)
            benchmark.print_results(fps_results)
            print(f"\n✓ Achieved {fps_results['fps']:.1f} FPS (target: 45 FPS)")
            if fps_results['fps'] >= 45:
                print("✓ Target FPS achieved for real-time underwater vehicle integration")
    
    if args.benchmark_fps and not args.convert_onnx and not args.convert_tensorrt:
        print("\n[5/6] Benchmarking PyTorch FPS...")
        pytorch_engine = InferenceEngine(
            str(output_dir / 'models' / f'{best_model_name.lower().replace(" ", "_")}_best.pth'),
            device=device,
            model_type='pytorch'
        )
        benchmark = FPSBenchmark(pytorch_engine, img_size=(args.img_size, args.img_size))
        fps_results = benchmark.benchmark(num_iterations=1000)
        benchmark.print_results(fps_results)
    
    print("\n[6/6] Generating Training Plots...")
    for model_name, model_info in results.items():
        if 'history' in model_info and len(model_info['history']['train_loss']) > 0:
            history = model_info['history']
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            axes[0, 0].plot(history['train_loss'], label='Train Loss')
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(history['train_acc'], label='Train Acc')
            axes[0, 1].plot(history['val_acc'], label='Val Acc')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            if 'lr' in history:
                axes[1, 0].plot(history['lr'])
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True)
            
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(
                output_dir / 'plots' / f'{model_name.lower().replace(" ", "_")}_training_history.png',
                dpi=300, bbox_inches='tight'
            )
            plt.close()
    
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'img_size': args.img_size,
        'use_tta': args.use_tta,
        'use_ensemble': args.use_ensemble,
        'use_focal_loss': args.use_focal_loss,
        'use_label_smoothing': args.use_label_smoothing,
        'use_lookahead': args.use_lookahead,
        'warmup_epochs': args.warmup_epochs,
        'best_model': best_model_name,
        'best_accuracy': float(evaluation_results[best_model_name]['accuracy']),
        'best_map': float(evaluation_results[best_model_name]['mean_ap']),
        'improvement_over_baseline': float(improvement_over_baseline)
    }
    save_config(config, output_dir / 'training_config.json')
    
    print("\n" + "=" * 70)
    print("Training and Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Best model: {best_model_name}")
    print(f"  Accuracy: {evaluation_results[best_model_name]['accuracy']*100:.2f}%")
    print(f"  mAP: {evaluation_results[best_model_name]['mean_ap']*100:.2f}%")
    print(f"  Top-5 Accuracy: {evaluation_results[best_model_name]['top5_accuracy']*100:.2f}%")
    if improvement_over_baseline > 0:
        print(f"  Improvement over ResNet-50: {improvement_over_baseline:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
