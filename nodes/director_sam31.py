"""Internal queue bridge; importing the plugin does not import SAM."""


class DirectorSAM31Job:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job_id": ("STRING", {"default": ""})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    DEV_ONLY = True
    CATEGORY = "H3_D_NEO/internal"
    DESCRIPTION = "Internal SAM3.1 queue job. Use the Director drawer."

    def execute(self, job_id):
        from ..director.sam31_routes import execute_job
        execute_job(str(job_id))
        return (str(job_id),)
