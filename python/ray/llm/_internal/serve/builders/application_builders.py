from ray._common.utils import import_attr

from ray.llm._internal.serve.deployments.routers.prefix_tree_deployment import PrefixTree

from typing import List, Optional, Sequence

from ray.serve.deployment import Application
from ray.serve.handle import DeploymentHandle

from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.deployments.llm.llm_server import LLMDeployment
from ray.llm._internal.serve.configs.server_models import (
    LLMConfig,
    LLMServingArgs,
    LLMEngine,
)
from ray.llm._internal.serve.deployments.routers.router import (
    LLMRouter,
)

logger = get_logger(__name__)


def build_llm_deployment(
    llm_config: LLMConfig,
    deployment_kwargs: Optional[dict] = None,
) -> Application:
    # print(f"[application_builders.py: build_llm_deployment] llm_config: {llm_config} and deployment_kwargs: {deployment_kwargs}")
    if deployment_kwargs is None:
        deployment_kwargs = {}

    deployment_options = llm_config.get_serve_options(
        name_prefix="LLMDeployment:",
    )

    # if llm_config.replica_scheduler_cls_path:
    #     replica_scheduler_cls = import_attr(llm_config.replica_scheduler_cls_path)
    #     deployment_options["replica_scheduler_cls"] = replica_scheduler_cls

    # print(f"[application_builders.py: build_llm_deployment] binding llm_config: {llm_config}")
    return LLMDeployment.options(**deployment_options).bind(
        llm_config=llm_config, **deployment_kwargs
    )


def _get_llm_deployments(
    llm_base_models: Optional[Sequence[LLMConfig]] = None,
    deployment_kwargs: Optional[dict] = None,
) -> List[DeploymentHandle]:
    llm_deployments = []
    for llm_config in llm_base_models:
        if llm_config.llm_engine == LLMEngine.vLLM:
            llm_deployments.append(build_llm_deployment(llm_config, deployment_kwargs))
        else:
            # Note (genesu): This should never happen because we validate the engine
            # in the config.
            raise ValueError(f"Unsupported engine: {llm_config.llm_engine}")

    return llm_deployments

from ray import serve
def build_openai_app(llm_serving_args: LLMServingArgs) -> Application:
    # print(f"[application_builders.py: build_openai_app] llm_serving_args: {llm_serving_args}")
    rayllm_args = LLMServingArgs.model_validate(llm_serving_args).parse_args()

    llm_configs = rayllm_args.llm_configs
    model_ids = {m.model_id for m in llm_configs}
    if len(model_ids) != len(llm_configs):
        raise ValueError("Duplicate models found. Make sure model ids are unique.")

    if len(llm_configs) == 0:
        logger.error(
            "List of models is empty. Maybe some parameters cannot be parsed into the LLMConfig config."
        )
    tree_deployment = PrefixTree.bind()
    serve.run(tree_deployment)
    # print(f"[application_builders.py: build_openai_app] tree_deployment: {tree_deployment}")
    llm_deployments = _get_llm_deployments(llm_configs)
    # print(f"[application_builders.py: build_openai_app] llm_deployments: {llm_deployments}")
    return LLMRouter.as_deployment(llm_configs=llm_configs).options(autoscaling_config=dict(min_replicas=1, max_replicas=1, initial_replicas=1)).bind(
        llm_deployments=llm_deployments, 
        tree_deployment=tree_deployment
    )
