"""Regenerate Blender scripts from the original task and exact supervised CoT."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import argparse
import ast
import base64
import mimetypes
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request

from PIL import Image

CASES = ("Comforter_032__image-to-3d", "Auger_001__text-to-3d",
         "TaroMonocot_024__text-to-3d", "AquariumTank_000__text-to-3d")
FENCE = re.compile(r"^\s*" + chr(96)*3 + r"(?:python|py)?\s*\n(.*?)\n" + chr(96)*3 + r"\s*$",re.S | re.I)

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def attempt(row, case, endpoint, model, media_root, sandbox, max_tokens,
            inference_timeout=240, render_timeout=100, render_slots=None,
            reject_truncation=False, quiet=False):
    identity=row["id"]
    text=row["messages"][2]["reasoning_content"]
    case.mkdir(parents=True)
    (case/"cot.txt").write_bytes(text.encode("utf-8"))
    result={"id":identity,"cot_sha256":sha(text),"cot_lines":len(text.splitlines()),"model":model,
            "status":"inference_failed"}

    def finish(status, **detail):
        result.update(status=status,**detail)
        write(case/"result.json",result)
        if not quiet:
            print(json.dumps(result,ensure_ascii=False),flush=True)
        return result

    system, user, _ = row["messages"]
    write(case/"original_prompt.json",[system,user])
    plan = "\n\n[Supplied implementation plan]\n" + text
    original = user["content"]
    num_images = 0
    if isinstance(original,str):
        combined = original + plan
    else:
        combined = []
        for block in original:
            if block["type"]=="text":
                combined.append(block)
            elif block["type"]=="image":
                path = (media_root/block["path"]).resolve()
                if not path.is_relative_to(media_root) or not path.is_file():
                    return finish("input_media_unavailable",detail=block["path"])
                mime=mimetypes.guess_type(path.name)[0] or "image/webp"
                encoded=base64.b64encode(path.read_bytes()).decode("ascii")
                combined.append({"type":"image_url","image_url":{"url":f"data:{mime};base64,{encoded}"}})
                num_images+=1
            else:
                return finish("unsupported_input_block",detail=block["type"])
        combined.append({"type":"text","text":plan})
    result["original_prompt_sha256"]=sha(json.dumps([system,user],ensure_ascii=False))
    result["reference_images_sent"]=num_images
    body={"model":model,"messages":[system,{"role":"user","content":combined}],
          "temperature":0,"max_tokens":max_tokens,"chat_template_kwargs":{"enable_thinking":False}}
    start=time.monotonic()
    try:
        request=urllib.request.Request(endpoint+"/chat/completions",data=json.dumps(body).encode(),
                                       headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(request,timeout=inference_timeout) as response:
            reply=json.load(response)
    except urllib.error.HTTPError as error:
        return finish("inference_failed",http_status=error.code,detail=error.read().decode("utf-8",errors="replace")[:3000])
    except (OSError,ValueError) as error:
        return finish("inference_failed",detail=str(error))
    result["inference_seconds"]=round(time.monotonic()-start,2)
    first=reply["choices"][0]
    result["finish_reason"]=first["finish_reason"]
    result["usage"]=reply.get("usage")
    answer=first["message"].get("content") or ""
    (case/"response.txt").write_bytes(answer.encode("utf-8"))
    if reject_truncation and result["finish_reason"] == "length":
        return finish("output_limit")
    if not answer.strip():
        return finish("empty_model_code")
    source=answer
    fenced=FENCE.fullmatch(source)
    if fenced:
        source=fenced[1]
    result["unwrapped_single_code_fence"]=bool(fenced)
    result["code_lines"]=len(source.splitlines())
    result["code_sha256"]=sha(source)
    (case/"asset.py").write_bytes(source.encode("utf-8"))
    try:
        ast.parse(source)
    except SyntaxError as error:
        return finish("python_syntax_error",detail=f"{error.msg}, line {error.lineno}")

    for child in ("raw",".root"):
        (case/child).mkdir()
    write(case/"config.json",{"size":512,"threads":4,
         "views":[[1.7,-2.5,1.4],[.25,-1,.4]] if identity.startswith("Auger") else [[1.7,-2.5,1.4],[.25,-.2,1]]})
    command=["unshare","-Urmpnf","--kill-child","/usr/bin/sh",str(sandbox/"isolated_blender.sh"),
             str(case),str(Path.home()/"opt/blender-5.0.1-linux-x64"),str(sandbox/"batch_blender_worker.py"),""]
    env={"PATH":"/usr/bin:/bin","CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"1"}
    with (render_slots or nullcontext()), (case/"blender.log").open("wb") as log:
        start=time.monotonic()
        process=subprocess.Popen(command,cwd=case,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        try:
            result["blender_exit_code"]=process.wait(timeout=render_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            return finish("blender_timeout")
    result["blender_seconds"]=round(time.monotonic()-start,2)
    metric=case/"raw/metrics.json"
    if result["blender_exit_code"]!=0 or not metric.is_file() or metric.is_symlink():
        log=(case/"blender.log").read_text(encoding="utf-8",errors="replace")
        code_error = (result["blender_exit_code"] == 12 and 'File "/input/asset.py"' in log)
        status = "blender_error" if code_error else "blender_environment_error"
        return finish(status,detail="\n".join(log.splitlines()[-12:]))
    metrics=json.loads(metric.read_bytes())
    result.update({key:metrics.get(key) for key in ("objects","vertices","faces","blender","script_completed")})
    if (not result["script_completed"] or not result["faces"] or metrics.get("nonfinite_vertices")
            or metrics.get("invalid_bounds")):
        return finish("invalid_mesh")
    views=[]
    try:
        for filename in ("view-1.png","view-2.png"):
            view = case/"raw"/filename
            if not view.is_file() or view.is_symlink():
                return finish("render_error",detail=f"Missing or invalid artifact: {filename}")
            with Image.open(view) as image:
                alpha=image.getchannel("A")
                views.append(round(100*sum(alpha.histogram()[32:])/(image.width*image.height),3))
    except (OSError,ValueError) as error:
        return finish("render_error",detail=str(error))
    return finish("pass" if all(views) else "empty_render",view_foreground_pct=views)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--sandbox",required=True,type=Path)
    p.add_argument("--model",default="qwen3_8_27b_native_cot_label")
    p.add_argument("--endpoint",default="http://127.0.0.1:8001/v1")
    p.add_argument("--max-tokens",type=int,default=6144)
    p.add_argument("--dataset",type=Path,default=Path.home()/"sft_Data/sft_train.jsonl")
    p.add_argument("--id",action="append",dest="ids")
    p.add_argument("--media-root",type=Path,default=Path.home()/"sft_set")
    args=p.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    requested=tuple(args.ids or CASES)
    rows={}
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        if row["id"] in requested:
            rows[row["id"]]=row
    if set(requested)!=set(rows):
        p.error("Some sample ID was not found in the dataset")
    sandbox=args.sandbox.resolve()
    endpoint=args.endpoint.rstrip("/")
    with urllib.request.urlopen(endpoint+"/models",timeout=15) as response:
        model_list=json.load(response)["data"]
    model_entry=next(item for item in model_list if item["id"]==args.model)
    config=Path(model_entry.get("root") or ".")/"config.json"
    architecture={}
    if config.is_file():
        settings=json.loads(config.read_bytes())
        architecture={key:settings.get(key) for key in ("architectures","model_type","_name_or_path") if key in settings}
    write(out/"run.json",{"model":args.model,"model_entry":model_entry,"architecture":architecture,"endpoint":endpoint,
                          "dataset":str(args.dataset),"input":"original system + original user + assistant.reasoning_content",
                          "media_root":str(args.media_root),"temperature":0,"max_tokens":args.max_tokens,"enable_thinking":False})
    print("MODEL",args.model,architecture,flush=True)

    def evaluate(row):
        return attempt(row, out/row["id"], endpoint, args.model, args.media_root.resolve(),
                       sandbox, args.max_tokens)

    with ThreadPoolExecutor(max_workers=min(4,len(requested))) as pool:
        results=list(pool.map(lambda i:evaluate(rows[i]),requested))
    write(out/"summary.json",{"model":args.model,"architecture":architecture,
                              "passed":sum(x["status"]=="pass" for x in results),"total":len(results),"cases":results})
    print("DONE",out,flush=True)

if __name__=="__main__":
    main()
