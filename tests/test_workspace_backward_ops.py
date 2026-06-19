from __future__ import annotations

import numpy as np
import pytest

from tensionforge.ops import (
    batched_dot_backward_device, broadcast_rows_backward_device,
    concatenate_backward_device, cross_entropy_device, embedding_backward_device,
    gather_backward_device, gelu_backward_device, layer_norm_backward_device,
    row_delta_norm_backward_device, scatter_backward_device, softmax_backward_device,
    tension_penalty_backward_device, tension_update_broadcast_backward_device,
    weighted_sum_backward_device, workspace_reduce_backward_device,
)
from tensionforge.runtime import TensionForgeRuntime
from tensionforge.tensor import DeviceTensor


torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def runtime():
    return TensionForgeRuntime()


def d(runtime, value, dtype=np.float32):
    return DeviceTensor.from_numpy(runtime, np.asarray(value, dtype=dtype))


def close(actual, expected, atol=2e-5):
    np.testing.assert_allclose(actual.to_numpy(), expected.detach().numpy(), rtol=2e-5, atol=atol)


def test_layer_norm_gelu_softmax_backward(runtime):
    rng=np.random.default_rng(2); x=torch.tensor(rng.normal(size=(3,7)).astype(np.float32),requires_grad=True)
    w=torch.tensor(rng.normal(size=7).astype(np.float32),requires_grad=True); b=torch.tensor(rng.normal(size=7).astype(np.float32),requires_grad=True)
    upstream=torch.tensor(rng.normal(size=(3,7)).astype(np.float32))
    y=torch.nn.functional.layer_norm(x,(7,),w,b,1e-5); y.backward(upstream)
    gx,gw,gb=layer_norm_backward_device(runtime,d(runtime,x.detach()),d(runtime,w.detach()),d(runtime,upstream))
    close(gx,x.grad);close(gw,w.grad);close(gb,b.grad)
    for fn, tf in [(gelu_backward_device,lambda z: torch.nn.functional.gelu(z)),(softmax_backward_device,None)]:
        z=torch.tensor(rng.normal(size=(3,7)).astype(np.float32),requires_grad=True); g=torch.tensor(rng.normal(size=(3,7)).astype(np.float32))
        if tf: out=tf(z); out.backward(g); got=fn(runtime,d(runtime,z.detach()),d(runtime,g))
        else: out=z.softmax(-1);out.backward(g);got=fn(runtime,d(runtime,out.detach()),d(runtime,g))
        close(got,z.grad)


def test_reductions_gather_scatter_and_broadcast(runtime):
    rng=np.random.default_rng(3); raw=rng.normal(size=(2,5,4)).astype(np.float32)
    raw += np.arange(5,dtype=np.float32)[None,:,None]*0.2
    x=torch.tensor(raw,requires_grad=True); gm=torch.tensor(rng.normal(size=(2,4)).astype(np.float32)); gx=torch.tensor(rng.normal(size=(2,4)).astype(np.float32))
    (x.mean(1)*gm+x.max(1).values*gx).sum().backward()
    got=workspace_reduce_backward_device(runtime,d(runtime,raw),d(runtime,gm),d(runtime,gx));close(got,x.grad)
    idx=torch.tensor([[4,1],[0,3]],dtype=torch.long); x.grad=None; gathered=x.gather(1,idx[...,None].expand(-1,-1,4));g=torch.tensor(rng.normal(size=(2,2,4)).astype(np.float32));gathered.backward(g)
    got=gather_backward_device(runtime,d(runtime,g),d(runtime,idx.numpy(),np.int32),raw.shape);close(got,x.grad)
    base=torch.tensor(raw,requires_grad=True);u=torch.tensor(rng.normal(size=(2,2,4)).astype(np.float32),requires_grad=True);out=base.scatter(1,idx[...,None].expand(-1,-1,4),u);go=torch.tensor(rng.normal(size=raw.shape).astype(np.float32));out.backward(go)
    gi,gu=scatter_backward_device(runtime,d(runtime,go),d(runtime,idx.numpy(),np.int32));close(gi,base.grad);close(gu,u.grad)
    z=torch.tensor(rng.normal(size=(2,4)).astype(np.float32),requires_grad=True);br=z[:,None,:].expand(2,3,4);bg=torch.tensor(rng.normal(size=(6,4)).astype(np.float32));br.reshape(6,4).backward(bg)
    close(broadcast_rows_backward_device(runtime,d(runtime,bg),2,3),z.grad)


def test_attention_cell_norm_embedding_and_loss_backward(runtime):
    rng=np.random.default_rng(4); b,n,f=2,4,3
    q=torch.tensor(rng.normal(size=(b,f)).astype(np.float32),requires_grad=True);v=torch.tensor(rng.normal(size=(b,n,f)).astype(np.float32),requires_grad=True);g=torch.tensor(rng.normal(size=(b,n)).astype(np.float32));scale=f**-0.5
    ((q[:,None,:]*v).sum(-1)*scale).backward(g);gq,gv=batched_dot_backward_device(runtime,d(runtime,q.detach()),d(runtime,v.detach()),d(runtime,g),scale);close(gq,q.grad);close(gv,v.grad)
    w=torch.tensor(rng.normal(size=(b,n)).astype(np.float32),requires_grad=True);v.grad=None;go=torch.tensor(rng.normal(size=(b,f)).astype(np.float32));(w[...,None]*v).sum(1).backward(go);gw,gv=weighted_sum_backward_device(runtime,d(runtime,w.detach()),d(runtime,v.detach()),d(runtime,go));close(gw,w.grad);close(gv,v.grad)
    a=torch.tensor(rng.normal(size=(5,f)).astype(np.float32),requires_grad=True);bb=torch.tensor(rng.normal(size=(5,f)).astype(np.float32),requires_grad=True);ng=torch.tensor(rng.normal(size=(5,1)).astype(np.float32));y=(a-bb).norm(dim=-1,keepdim=True);y.backward(ng);ga,gb=row_delta_norm_backward_device(runtime,d(runtime,a.detach()),d(runtime,bb.detach()),d(runtime,y.detach()),d(runtime,ng));close(ga,a.grad);close(gb,bb.grad)
    s=torch.tensor(rng.normal(size=(5,f)).astype(np.float32),requires_grad=True);p=torch.tensor(rng.normal(size=(5,f)).astype(np.float32),requires_grad=True);t=torch.sigmoid(torch.tensor(rng.normal(size=(5,1)).astype(np.float32))).detach().requires_grad_();ug=torch.tensor(rng.normal(size=(5,f)).astype(np.float32));(s+t*(p-s)).backward(ug);gs,gp,gt=tension_update_broadcast_backward_device(runtime,d(runtime,s.detach()),d(runtime,p.detach()),d(runtime,t.detach()),d(runtime,ug));close(gs,s.grad);close(gp,p.grad);close(gt,t.grad)
    pen=torch.tensor(np.linspace(-.1,1.1,b*n,dtype=np.float32).reshape(b,n),requires_grad=True);pg=torch.tensor(rng.normal(size=(b,n)).astype(np.float32));torch.log1p(-pen.clamp(0,.999)).backward(pg);_,gpen=tension_penalty_backward_device(runtime,d(runtime,pen.detach()),d(runtime,pg));close(gpen,pen.grad,5e-4)
    ids=np.array([[1,2,1],[0,2,3]],np.int32);eg=torch.tensor(rng.normal(size=(2,3,f)).astype(np.float32));table=torch.tensor(rng.normal(size=(5,f)).astype(np.float32),requires_grad=True);torch.nn.functional.embedding(torch.tensor(ids,dtype=torch.long),table).backward(eg);close(embedding_backward_device(runtime,d(runtime,ids,np.int32),d(runtime,eg),(5,f)),table.grad)
    logits=torch.tensor(rng.normal(size=(4,6)).astype(np.float32),requires_grad=True);targets=np.array([1,5,0,3],np.int32);loss=torch.nn.functional.cross_entropy(logits,torch.tensor(targets,dtype=torch.long));loss.backward();losses,cg=cross_entropy_device(runtime,d(runtime,logits.detach()),d(runtime,targets,np.int32));np.testing.assert_allclose(losses.to_numpy().sum(),loss.item(),rtol=2e-5);close(cg,logits.grad)


def test_concatenate_backward(runtime):
    g=np.arange(30,dtype=np.float32).reshape(3,10);left,right=concatenate_backward_device(runtime,d(runtime,g),4)
    np.testing.assert_array_equal(left.to_numpy(),g[:,:4]);np.testing.assert_array_equal(right.to_numpy(),g[:,4:])
