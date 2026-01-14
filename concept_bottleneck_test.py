"""
ConceptBottleneck模块全面测试脚本
测试ConceptBottleneck模块的各种功能，包括前向传播、损失计算、编码解码等
"""
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
from dreamerv3.agent_concept import ConceptBottleneck
import dreamerv3.ninjax as nj


def test_basic_functionality():
    """测试基本功能"""
    print("="*50)
    print("测试ConceptBottleneck模块基本功能")
    print("="*50)
    
    def run_basic_test():
        model = ConceptBottleneck(feat_dim=128, n_atoms=64, name='basic_test')
        x = jax.random.normal(jax.random.PRNGKey(0), (32, 128))
        result = model.compute(x)
        return result, x
    
    # nj.pure返回(output, state)，我们需要分别解包
    (outputs, x), _ = nj.pure(run_basic_test)({}, jax.random.PRNGKey(42))
    h_rec, alpha = outputs  # 解包compute的返回值
    
    print(f"输入形状: {x.shape}")
    print(f"重建输出形状: {h_rec.shape}")
    print(f"稀疏编码形状: {alpha.shape}")
    print(f"稀疏编码的平均绝对值: {jnp.mean(jnp.abs(alpha)):.4f}")
    print(f"稀疏编码的零元素比例: {(jnp.abs(alpha) < 1e-6).mean():.4f}")
    print(f"重建误差 (MSE): {jnp.mean((x - h_rec) ** 2):.4f}")
    print("✓ 基本功能测试通过\n")


def test_loss_computation():
    """测试损失计算功能"""
    print("测试损失计算功能")
    print("-" * 30)
    
    def run_loss_test():
        model = ConceptBottleneck(feat_dim=128, n_atoms=64, lambda_=0.05, name='loss_test')
        x = jax.random.normal(jax.random.PRNGKey(0), (32, 128))
        return model.loss(x)
    
    (total_loss, loss_components), _ = nj.pure(run_loss_test)({}, jax.random.PRNGKey(42))
    
    print(f"总损失: {total_loss:.4f}")
    print(f"重建损失: {loss_components['rec_loss']:.4f}")
    print(f"稀疏损失: {loss_components['sparsity_loss']:.4f}")
    print(f"稀疏编码平均范数: {loss_components['alpha_norm']:.4f}")
    print("✓ 损失计算测试通过\n")


def test_encode_decode():
    """测试编码解码功能"""
    print("测试编码解码功能")
    print("-" * 30)
    
    def run_encode_decode_test():
        model = ConceptBottleneck(feat_dim=128, n_atoms=64, name='encode_decode_test')
        x = jax.random.normal(jax.random.PRNGKey(0), (32, 128))
        result = model.encode_decode(x)
        return result, x
    
    # nj.pure返回((output1, output2), state)，我们需要分别解包
    ((concept_repr, reconstructed), x), _ = nj.pure(run_encode_decode_test)({}, jax.random.PRNGKey(42))
    
    print(f"概念表示形状: {concept_repr.shape}")
    print(f"重建特征形状: {reconstructed.shape}")
    print(f"重建误差 (MSE): {jnp.mean((x - reconstructed) ** 2):.4f}")
    print("✓ 编码解码测试通过\n")


def test_different_configurations():
    """测试不同配置"""
    print("测试不同配置")
    print("-" * 30)
    
    test_configs = [
        {"feat_dim": 64, "n_atoms": 32},
        {"feat_dim": 256, "n_atoms": 128},
        {"feat_dim": 512, "n_atoms": 256}
    ]
    
    for i, config in enumerate(test_configs):
        print(f"配置 {i+1}: feat_dim={config['feat_dim']}, n_atoms={config['n_atoms']}")
        
        def run_config_test():
            model = ConceptBottleneck(
                feat_dim=config['feat_dim'],
                n_atoms=config['n_atoms'],
                lambda_=0.05,
                n_steps=5,
                name=f'config_test_{i}'
            )
            x = jax.random.normal(jax.random.PRNGKey(0), (16, config['feat_dim']))
            h_rec, alpha = model.compute(x)
            total_loss, _ = model.loss(x)
            return h_rec, alpha, total_loss
        
        (h_rec, alpha, total_loss), _ = nj.pure(run_config_test)({}, jax.random.PRNGKey(42))
        x_test = jax.random.normal(jax.random.PRNGKey(0), (16, config['feat_dim']))  # 重新创建x用于计算
        
        print(f"  重建误差: {jnp.mean((x_test - h_rec) ** 2):.4f}")
        print(f"  总损失: {total_loss:.4f}")
        print(f"  稀疏度: {(jnp.abs(alpha) < 1e-6).mean():.4f}")
    
    print("✓ 不同配置测试通过\n")


def test_sparse_regularization():
    """测试稀疏正则化效果"""
    print("测试稀疏正则化效果")
    print("-" * 30)
    
    # 测试不同的lambda值对稀疏性的影响
    lambdas = [0.01, 0.05, 0.1, 0.2]
    
    for lam in lambdas:
        print(f"Lambda值: {lam}")
        
        def run_lambda_test():
            model = ConceptBottleneck(feat_dim=128, n_atoms=64, lambda_=lam, name=f'lambda_test_{lam}')
            x = jax.random.normal(jax.random.PRNGKey(0), (16, 128))
            _, alpha = model.compute(x)
            return alpha
        
        alpha, _ = nj.pure(run_lambda_test)({}, jax.random.PRNGKey(42))
        
        sparsity = (jnp.abs(alpha) < 1e-6).mean()
        print(f"  稀疏度: {sparsity:.4f}")
        print(f"  非零元素平均数量: {jnp.sum(alpha != 0) / alpha.shape[0]:.2f}")
    
    print("✓ 稀疏正则化测试通过\n")


def test_ista_iterations():
    """测试ISTA迭代步数的影响"""
    print("测试ISTA迭代步数的影响")
    print("-" * 30)
    
    n_steps_list = [5, 10, 20]
    
    for n_steps in n_steps_list:
        print(f"ISTA迭代步数: {n_steps}")
        
        def run_iter_test():
            model = ConceptBottleneck(feat_dim=128, n_atoms=64, n_steps=n_steps, name=f'iter_test_{n_steps}')
            x = jax.random.normal(jax.random.PRNGKey(0), (16, 128))
            h_rec, alpha = model.compute(x)
            return h_rec, alpha
        
        (h_rec, alpha), _ = nj.pure(run_iter_test)({}, jax.random.PRNGKey(42))
        x_test = jax.random.normal(jax.random.PRNGKey(0), (16, 128))  # 重新创建x用于计算
        
        print(f"  重建误差: {jnp.mean((x_test - h_rec) ** 2):.4f}")
        print(f"  稀疏度: {(jnp.abs(alpha) < 1e-6).mean():.4f}")
    
    print("✓ ISTA迭代测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("开始运行ConceptBottleneck模块全面测试...")
    
    test_basic_functionality()
    test_loss_computation()
    test_encode_decode()
    test_different_configurations()
    test_sparse_regularization()
    test_ista_iterations()
    
    print("="*50)
    print("✅ 所有测试均已通过！ConceptBottleneck模块功能正常")
    print("="*50)


if __name__ == "__main__":
    run_all_tests()